import 'dart:io';
import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;

/// AI Bin Fill Level Detector
/// Uses edge detection and region analysis to estimate bin fill percentage
/// 
/// Research Paper Reference:
/// - Algorithm: Vertical Region Analysis + Edge Density Calculation
/// - Features: Top-down fill estimation, empty space detection
/// - This detects how full a waste bin is from a photo
class BinFillDetectorService {
  static bool _isInitialized = false;

  /// Initialize the service
  static Future<bool> initialize() async {
    _isInitialized = true;
    if (kDebugMode) {
      print('BinFillDetector: Initialized with region analysis');
    }
    return true;
  }

  /// Detect bin fill level from image path (mobile/desktop)
  static Future<BinFillResult> detectFillLevel(String imagePath) async {
    if (!_isInitialized) await initialize();

    final stopwatch = Stopwatch()..start();

    try {
      // Web platform check
      if (kIsWeb) {
        return BinFillResult(
          fillPercentage: 0,
          status: BinStatus.unknown,
          analysisDetails: 'Please use detectFillLevelFromBytes for web platform',
          processingTimeMs: stopwatch.elapsedMilliseconds,
        );
      }
      
      final file = File(imagePath);
      if (!await file.exists()) {
        return BinFillResult(
          fillPercentage: 0,
          status: BinStatus.unknown,
          analysisDetails: 'Image not found at path: $imagePath',
          processingTimeMs: 0,
        );
      }

      final bytes = await file.readAsBytes();
      return _processImageBytes(bytes, stopwatch);
    } catch (e, stackTrace) {
      stopwatch.stop();
      if (kDebugMode) {
        print('BinFillDetector Error: $e');
        print('Stack trace: $stackTrace');
      }
      return BinFillResult(
        fillPercentage: 0,
        status: BinStatus.unknown,
        analysisDetails: 'Error processing image: $e',
        processingTimeMs: stopwatch.elapsedMilliseconds,
      );
    }
  }

  /// Detect bin fill level from image bytes (works on all platforms including web)
  static Future<BinFillResult> detectFillLevelFromBytes(Uint8List bytes) async {
    if (!_isInitialized) await initialize();
    final stopwatch = Stopwatch()..start();
    return _processImageBytes(bytes, stopwatch);
  }

  /// Process image bytes and return fill result
  static Future<BinFillResult> _processImageBytes(Uint8List bytes, Stopwatch stopwatch) async {
    try {
      final image = img.decodeImage(bytes);

      if (image == null) {
        return BinFillResult(
          fillPercentage: 0,
          status: BinStatus.unknown,
          analysisDetails: 'Could not decode image. Supported formats: JPEG, PNG, GIF, BMP, TIFF, WebP',
          processingTimeMs: stopwatch.elapsedMilliseconds,
        );
      }

      if (kDebugMode) {
        print('BinFillDetector: Image decoded - ${image.width}x${image.height}');
      }

      // Resize for faster processing
      final resized = img.copyResize(image, width: 200, height: 300);
      
      // Analyze vertical regions (top to bottom)
      final result = await _analyzeVerticalFill(resized);
      
      stopwatch.stop();

      if (kDebugMode) {
        print('BinFillDetector: Analysis complete - ${result.fillPercentage}% fill');
      }

      return BinFillResult(
        fillPercentage: result.fillPercentage,
        status: _getStatus(result.fillPercentage),
        emptySpaceTop: result.emptySpaceRatio,
        contentDensity: result.contentDensity,
        analysisDetails: _generateDetails(result),
        processingTimeMs: stopwatch.elapsedMilliseconds,
        recommendation: _getRecommendation(result.fillPercentage),
      );
    } catch (e, stackTrace) {
      stopwatch.stop();
      if (kDebugMode) {
        print('BinFillDetector Processing Error: $e');
        print('Stack trace: $stackTrace');
      }
      return BinFillResult(
        fillPercentage: 0,
        status: BinStatus.unknown,
        analysisDetails: 'Error analyzing image: $e',
        processingTimeMs: stopwatch.elapsedMilliseconds,
      );
    }
  }

  /// Analyze vertical fill using region-based approach with improved algorithm
  static Future<_FillAnalysis> _analyzeVerticalFill(img.Image image) async {
    final width = image.width;
    final height = image.height;
    
    // Divide image into horizontal strips (top to bottom)
    const int numStrips = 10;
    final stripHeight = height ~/ numStrips;
    
    List<double> stripDensities = [];
    List<double> stripEdgeCounts = [];
    List<double> stripTextures = [];
    
    // Calculate global image statistics for normalization
    double globalBrightnessSum = 0;
    int totalPixels = 0;
    
    // First pass: calculate global statistics
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final pixel = image.getPixel(x, y);
        globalBrightnessSum += (pixel.r + pixel.g + pixel.b) / 3;
        totalPixels++;
      }
    }
    final globalMeanBrightness = globalBrightnessSum / totalPixels;
    
    // Analyze each strip
    for (int strip = 0; strip < numStrips; strip++) {
      final startY = strip * stripHeight;
      final endY = min((strip + 1) * stripHeight, height);
      
      double colorVariance = 0;
      double edgeCount = 0;
      double textureScore = 0;
      int pixelCount = 0;
      
      List<int> hues = [];
      List<double> brightnesses = [];
      
      for (int y = startY; y < endY; y++) {
        for (int x = 0; x < width; x++) {
          final pixel = image.getPixel(x, y);
          final r = pixel.r.toInt();
          final g = pixel.g.toInt();
          final b = pixel.b.toInt();
          
          // Convert to HSV
          final hsv = _rgbToHsv(r, g, b);
          hues.add(hsv[0].toInt());
          
          // Calculate brightness
          final brightness = (r + g + b) / 3.0;
          brightnesses.add(brightness);
          
          // Edge detection (Sobel-like gradient)
          if (x > 0 && y > startY && x < width - 1 && y < endY - 1) {
            final prevX = image.getPixel(x - 1, y);
            final nextX = image.getPixel(x + 1, y);
            final prevY = image.getPixel(x, y - 1);
            final nextY = image.getPixel(x, y + 1);
            
            // Sobel gradient approximation
            final gradX = ((nextX.r.toInt() - prevX.r.toInt()).abs() + 
                          (nextX.g.toInt() - prevX.g.toInt()).abs() + 
                          (nextX.b.toInt() - prevX.b.toInt()).abs()) / 3;
            final gradY = ((nextY.r.toInt() - prevY.r.toInt()).abs() + 
                          (nextY.g.toInt() - prevY.g.toInt()).abs() + 
                          (nextY.b.toInt() - prevY.b.toInt()).abs()) / 3;
            
            final gradient = sqrt(gradX * gradX + gradY * gradY);
            
            if (gradient > 30) {
              edgeCount++;
              textureScore += gradient;
            }
          }
          pixelCount++;
        }
      }
      
      // Calculate variance in this strip (color variance)
      if (hues.isNotEmpty) {
        final avgHue = hues.reduce((a, b) => a + b) / hues.length;
        colorVariance = hues.map((h) => pow(h - avgHue, 2)).reduce((a, b) => a + b) / hues.length;
      }
      
      // Calculate brightness variance (indicates content presence)
      double brightnessVariance = 0;
      if (brightnesses.isNotEmpty) {
        final avgBrightness = brightnesses.reduce((a, b) => a + b) / brightnesses.length;
        brightnessVariance = brightnesses.map((b) => pow(b - avgBrightness, 2)).reduce((a, b) => a + b) / brightnesses.length;
      }
      
      stripDensities.add(colorVariance + brightnessVariance);
      stripEdgeCounts.add(pixelCount > 0 ? edgeCount / pixelCount : 0);
      stripTextures.add(pixelCount > 0 ? textureScore / pixelCount : 0);
    }
    
    // Normalize strip values
    final maxDensity = stripDensities.reduce(max);
    final maxEdge = stripEdgeCounts.reduce(max);
    final maxTexture = stripTextures.reduce(max);
    
    if (maxDensity > 0) {
      stripDensities = stripDensities.map((d) => d / maxDensity).toList();
    }
    if (maxEdge > 0) {
      stripEdgeCounts = stripEdgeCounts.map((e) => e / maxEdge).toList();
    }
    if (maxTexture > 0) {
      stripTextures = stripTextures.map((t) => t / maxTexture).toList();
    }
    
    // Combined score for each strip
    List<double> combinedScores = [];
    for (int i = 0; i < numStrips; i++) {
      // Weight: edges 40%, texture 35%, color variance 25%
      final score = stripEdgeCounts[i] * 0.4 + stripTextures[i] * 0.35 + stripDensities[i] * 0.25;
      combinedScores.add(score);
    }
    
    // Find where content starts (from top)
    // Empty bin top has low combined score
    int emptyStripsFromTop = 0;
    final avgScore = combinedScores.reduce((a, b) => a + b) / combinedScores.length;
    final threshold = avgScore * 0.4; // 40% of average is considered empty
    
    for (int i = 0; i < numStrips; i++) {
      if (combinedScores[i] < threshold) {
        emptyStripsFromTop++;
      } else {
        break; // Content starts here
      }
    }
    
    // Calculate fill percentage with smoothing
    final emptyRatio = emptyStripsFromTop / numStrips;
    double fillPercentage = ((1 - emptyRatio) * 100);
    
    // Apply confidence adjustment based on overall texture
    final avgTexture = stripTextures.reduce((a, b) => a + b) / stripTextures.length;
    if (avgTexture < 0.2) {
      // Low texture overall - might be mostly empty
      fillPercentage = fillPercentage * 0.8;
    }
    
    // Adjust based on global brightness (very dark or very bright images need adjustment)
    if (globalMeanBrightness < 50 || globalMeanBrightness > 200) {
      // Image might be poorly lit, reduce confidence slightly
      fillPercentage = fillPercentage * 0.95;
    }
    
    fillPercentage = fillPercentage.clamp(5.0, 100.0);
    
    // Calculate content density (how packed the content is)
    final filledStrips = combinedScores.sublist(emptyStripsFromTop);
    final contentDensity = filledStrips.isNotEmpty 
        ? filledStrips.reduce((a, b) => a + b) / filledStrips.length
        : 0.0;
    
    return _FillAnalysis(
      fillPercentage: fillPercentage.roundToDouble(),
      emptySpaceRatio: emptyRatio,
      contentDensity: contentDensity.clamp(0.0, 1.0),
      stripAnalysis: combinedScores.map((s) => s * 100).toList(),
    );
  }

  /// Convert RGB to HSV
  static List<double> _rgbToHsv(int r, int g, int b) {
    final rf = r / 255;
    final gf = g / 255;
    final bf = b / 255;

    final maxC = [rf, gf, bf].reduce(max);
    final minC = [rf, gf, bf].reduce(min);
    final delta = maxC - minC;

    double h = 0;
    if (delta != 0) {
      if (maxC == rf) {
        h = 60 * (((gf - bf) / delta) % 6);
      } else if (maxC == gf) {
        h = 60 * (((bf - rf) / delta) + 2);
      } else {
        h = 60 * (((rf - gf) / delta) + 4);
      }
    }
    if (h < 0) h += 360;

    final s = maxC == 0 ? 0.0 : delta / maxC;
    final v = maxC;

    return [h, s, v];
  }

  /// Get bin status from fill percentage
  static BinStatus _getStatus(double fillPercentage) {
    if (fillPercentage >= 90) return BinStatus.overflowing;
    if (fillPercentage >= 75) return BinStatus.almostFull;
    if (fillPercentage >= 50) return BinStatus.halfFull;
    if (fillPercentage >= 25) return BinStatus.quarterFull;
    return BinStatus.empty;
  }

  /// Get recommendation based on fill level
  static String _getRecommendation(double fillPercentage) {
    if (fillPercentage >= 90) {
      return '🚨 Bin is overflowing! Schedule pickup immediately.';
    } else if (fillPercentage >= 75) {
      return '⚠️ Bin is almost full. Consider scheduling a pickup soon.';
    } else if (fillPercentage >= 50) {
      return '📦 Bin is half full. You can schedule pickup in 1-2 days.';
    } else if (fillPercentage >= 25) {
      return '✅ Bin has plenty of space. No immediate action needed.';
    } else {
      return '🌟 Bin is nearly empty. Great job managing your waste!';
    }
  }

  /// Generate analysis details
  static String _generateDetails(_FillAnalysis analysis) {
    final buffer = StringBuffer();
    buffer.writeln('=== BIN FILL ANALYSIS ===');
    buffer.writeln('Fill Level: ${analysis.fillPercentage.toStringAsFixed(1)}%');
    buffer.writeln('Empty Space (Top): ${(analysis.emptySpaceRatio * 100).toStringAsFixed(1)}%');
    buffer.writeln('Content Density: ${(analysis.contentDensity * 100).toStringAsFixed(1)}%');
    buffer.writeln('');
    buffer.writeln('Strip Analysis (Top to Bottom):');
    for (int i = 0; i < analysis.stripAnalysis.length; i++) {
      final density = analysis.stripAnalysis[i];
      final bar = '█' * (density / 100).clamp(0, 20).toInt();
      buffer.writeln('  ${i + 1}: $bar');
    }
    return buffer.toString();
  }
}

/// Fill analysis result
class _FillAnalysis {
  final double fillPercentage;
  final double emptySpaceRatio;
  final double contentDensity;
  final List<double> stripAnalysis;

  _FillAnalysis({
    required this.fillPercentage,
    required this.emptySpaceRatio,
    required this.contentDensity,
    required this.stripAnalysis,
  });
}

/// Bin status enum
enum BinStatus {
  empty,
  quarterFull,
  halfFull,
  almostFull,
  overflowing,
  unknown,
}

/// Bin fill detection result
class BinFillResult {
  final double fillPercentage;
  final BinStatus status;
  final double emptySpaceTop;
  final double contentDensity;
  final String analysisDetails;
  final int processingTimeMs;
  final String recommendation;

  BinFillResult({
    required this.fillPercentage,
    required this.status,
    this.emptySpaceTop = 0,
    this.contentDensity = 0,
    required this.analysisDetails,
    required this.processingTimeMs,
    this.recommendation = '',
  });

  String get fillPercentageText => '${fillPercentage.toStringAsFixed(0)}%';

  String get statusText {
    switch (status) {
      case BinStatus.empty:
        return 'Empty';
      case BinStatus.quarterFull:
        return 'Quarter Full';
      case BinStatus.halfFull:
        return 'Half Full';
      case BinStatus.almostFull:
        return 'Almost Full';
      case BinStatus.overflowing:
        return 'Overflowing!';
      case BinStatus.unknown:
        return 'Unknown';
    }
  }

  String get statusEmoji {
    switch (status) {
      case BinStatus.empty:
        return '🌟';
      case BinStatus.quarterFull:
        return '✅';
      case BinStatus.halfFull:
        return '📦';
      case BinStatus.almostFull:
        return '⚠️';
      case BinStatus.overflowing:
        return '🚨';
      case BinStatus.unknown:
        return '❓';
    }
  }
}

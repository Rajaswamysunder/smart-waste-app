import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

/// TensorFlow Lite Waste Classifier Service
/// Uses a trained MobileNetV2 model for waste classification
/// 
/// Model: waste_classifier.tflite (trained on TrashNet + custom data)
/// Classes: organic, recyclable, hazardous, ewaste, general
class TFLiteWasteClassifierService {
  static Interpreter? _interpreter;
  static List<String> _labels = [];
  static bool _isInitialized = false;
  static const int _inputSize = 224;
  static const String _modelPath = 'assets/models/waste_classifier.tflite';
  static const String _labelsPath = 'assets/models/waste_labels.txt';

  /// Initialize the TFLite interpreter
  static Future<bool> initialize() async {
    if (_isInitialized) return true;

    try {
      // Load model
      _interpreter = await Interpreter.fromAsset(_modelPath);
      
      // Load labels
      final labelsData = await rootBundle.loadString(_labelsPath);
      _labels = labelsData.split('\n').where((l) => l.isNotEmpty).toList();
      
      _isInitialized = true;
      
      if (kDebugMode) {
        print('TFLiteWasteClassifier: Initialized successfully');
        print('  Model: $_modelPath');
        print('  Labels: $_labels');
        print('  Input size: ${_inputSize}x$_inputSize');
      }
      
      return true;
    } catch (e) {
      if (kDebugMode) {
        print('TFLiteWasteClassifier: Failed to initialize - $e');
      }
      return false;
    }
  }

  /// Check if initialized
  static bool get isInitialized => _isInitialized;

  /// Classify waste from image file path
  static Future<WasteClassificationResult> classifyImage(String imagePath) async {
    if (!_isInitialized) {
      final success = await initialize();
      if (!success) {
        return WasteClassificationResult(
          label: 'General',
          confidence: 0.5,
          allPredictions: {'General': 0.5},
          analysisDetails: 'Model not initialized',
        );
      }
    }

    try {
      // Handle web platform
      if (kIsWeb) {
        return WasteClassificationResult(
          label: 'General',
          confidence: 0.5,
          allPredictions: {'General': 0.5},
          analysisDetails: 'TFLite not supported on web. Use classifyImageBytes instead.',
        );
      }

      final file = File(imagePath);
      if (!await file.exists()) {
        return WasteClassificationResult(
          label: 'Unknown',
          confidence: 0.0,
          allPredictions: {},
          analysisDetails: 'Image file not found',
        );
      }

      final bytes = await file.readAsBytes();
      return classifyImageBytes(bytes);
    } catch (e) {
      if (kDebugMode) {
        print('TFLiteWasteClassifier: Error - $e');
      }
      return WasteClassificationResult(
        label: 'General',
        confidence: 0.5,
        allPredictions: {'General': 0.5},
        analysisDetails: 'Error processing image: $e',
      );
    }
  }

  /// Classify waste from image bytes
  static Future<WasteClassificationResult> classifyImageBytes(Uint8List imageBytes) async {
    if (!_isInitialized) {
      final success = await initialize();
      if (!success) {
        return WasteClassificationResult(
          label: 'General',
          confidence: 0.5,
          allPredictions: {'General': 0.5},
          analysisDetails: 'Model not initialized',
        );
      }
    }

    try {
      final stopwatch = Stopwatch()..start();
      
      // Decode image
      final image = img.decodeImage(imageBytes);
      if (image == null) {
        return WasteClassificationResult(
          label: 'General',
          confidence: 0.5,
          allPredictions: {'General': 0.5},
          analysisDetails: 'Could not decode image',
        );
      }

      // Resize to model input size
      final resized = img.copyResize(image, width: _inputSize, height: _inputSize);
      
      // Convert to input tensor [1, 224, 224, 3]
      final input = _imageToInput(resized);
      
      // Output tensor [1, 5] (5 classes)
      final output = List.filled(1 * _labels.length, 0.0).reshape([1, _labels.length]);
      
      // Run inference
      _interpreter!.run(input, output);
      
      stopwatch.stop();
      
      // Process output
      final predictions = output[0] as List<double>;
      final scores = <String, double>{};
      
      for (int i = 0; i < _labels.length && i < predictions.length; i++) {
        // Map labels to display names
        final label = _mapLabelToDisplayName(_labels[i]);
        scores[label] = predictions[i];
      }
      
      // Find best prediction
      String bestLabel = 'General';
      double bestScore = 0.0;
      
      scores.forEach((label, score) {
        if (score > bestScore) {
          bestScore = score;
          bestLabel = label;
        }
      });
      
      // Generate analysis details
      final details = StringBuffer();
      details.writeln('=== AI Classification Result ===');
      details.writeln('Model: MobileNetV2 (TFLite)');
      details.writeln('Processing time: ${stopwatch.elapsedMilliseconds}ms');
      details.writeln('');
      details.writeln('Predictions:');
      
      // Sort by confidence
      final sortedScores = scores.entries.toList()
        ..sort((a, b) => b.value.compareTo(a.value));
      
      for (var entry in sortedScores) {
        final pct = (entry.value * 100).toStringAsFixed(1);
        details.writeln('  ${entry.key}: $pct%');
      }
      
      return WasteClassificationResult(
        label: bestLabel,
        confidence: bestScore,
        allPredictions: scores,
        analysisDetails: details.toString(),
        processingTimeMs: stopwatch.elapsedMilliseconds,
      );
    } catch (e, stackTrace) {
      if (kDebugMode) {
        print('TFLiteWasteClassifier: Inference error - $e');
        print('Stack trace: $stackTrace');
      }
      return WasteClassificationResult(
        label: 'General',
        confidence: 0.5,
        allPredictions: {'General': 0.5},
        analysisDetails: 'Inference error: $e',
      );
    }
  }

  /// Convert image to input tensor
  static List<List<List<List<double>>>> _imageToInput(img.Image image) {
    final input = List.generate(
      1,
      (_) => List.generate(
        _inputSize,
        (y) => List.generate(
          _inputSize,
          (x) {
            final pixel = image.getPixel(x, y);
            return [
              pixel.r / 255.0,  // Normalize to [0, 1]
              pixel.g / 255.0,
              pixel.b / 255.0,
            ];
          },
        ),
      ),
    );
    return input;
  }

  /// Map model label to display name
  static String _mapLabelToDisplayName(String label) {
    final mapping = {
      'organic': 'Organic',
      'recyclable': 'Recyclable',
      'hazardous': 'Hazardous',
      'ewaste': 'E-Waste',
      'general': 'General',
    };
    return mapping[label.toLowerCase()] ?? label;
  }

  /// Dispose interpreter
  static void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _isInitialized = false;
  }
}

/// Classification result
class WasteClassificationResult {
  final String label;
  final double confidence;
  final Map<String, double> allPredictions;
  final String analysisDetails;
  final int processingTimeMs;
  final bool isHumanDetected;

  WasteClassificationResult({
    required this.label,
    required this.confidence,
    required this.allPredictions,
    required this.analysisDetails,
    this.processingTimeMs = 0,
    this.isHumanDetected = false,
  });

  String get confidencePercentage => '${(confidence * 100).toStringAsFixed(1)}%';
  
  bool get isHighConfidence => confidence >= 0.7;
  
  String get emoji {
    switch (label) {
      case 'Organic':
        return '🌱';
      case 'Recyclable':
        return '♻️';
      case 'Hazardous':
        return '⚠️';
      case 'E-Waste':
        return '💻';
      case 'General':
        return '🗑️';
      default:
        return '❓';
    }
  }
  
  String get disposalTip {
    switch (label) {
      case 'Organic':
        return 'Place in green bin for composting. Food scraps, garden waste.';
      case 'Recyclable':
        return 'Place in blue bin. Ensure items are clean and dry.';
      case 'Hazardous':
        return 'Do NOT place in regular bins! Take to designated hazardous waste collection.';
      case 'E-Waste':
        return 'Take to e-waste collection center. Do not throw with regular waste.';
      case 'General':
        return 'Place in general waste bin. Cannot be recycled or composted.';
      default:
        return 'Unable to determine. Please manually classify.';
    }
  }
}

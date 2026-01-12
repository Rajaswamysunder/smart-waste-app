/// Stub implementation for TFLite Waste Classifier (for web)
/// Web platform doesn't support TFLite, so this returns dummy results

import 'dart:typed_data';

/// Stub classification result for web
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

/// Stub TFLite Waste Classifier Service for Web
/// Returns unavailable result since TFLite doesn't work on web
class TFLiteWasteClassifierService {
  static bool _isInitialized = false;
  
  /// Initialize - always returns false on web
  static Future<bool> initialize() async {
    return false;
  }

  /// Check if initialized
  static bool get isInitialized => _isInitialized;

  /// Classify waste from image file path - stub for web
  static Future<WasteClassificationResult> classifyImage(String imagePath) async {
    return WasteClassificationResult(
      label: 'General',
      confidence: 0.5,
      allPredictions: {'General': 0.5},
      analysisDetails: 'TFLite model not available on web platform. Use native app for AI classification.',
    );
  }

  /// Classify waste from image bytes - stub for web
  static Future<WasteClassificationResult> classifyImageBytes(Uint8List imageBytes) async {
    return WasteClassificationResult(
      label: 'General',
      confidence: 0.5,
      allPredictions: {'General': 0.5},
      analysisDetails: 'TFLite model not available on web platform. Use native app for AI classification.',
    );
  }

  /// Dispose - no-op for web
  static void dispose() {}
}

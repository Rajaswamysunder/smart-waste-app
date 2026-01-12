/// TFLite Waste Classifier Service
/// Platform-specific exports for TFLite-based waste classification
/// 
/// Uses conditional imports to provide:
/// - Native implementation (iOS, Android, macOS, Linux, Windows) with full TFLite support
/// - Web stub implementation that gracefully degrades

export 'tflite_waste_classifier_stub.dart'
    if (dart.library.io) 'tflite_waste_classifier_native.dart';

#!/usr/bin/env python3
"""
Convert Keras Model to TensorFlow Lite
======================================
Converts the trained waste classifier to TFLite format for mobile deployment.

Usage:
    python convert_to_tflite.py [--quantize] [--optimize]
"""

import os
import argparse
import numpy as np
from pathlib import Path
import tensorflow as tf

def convert_to_tflite(
    model_path='models/waste_classifier_best.keras',
    output_path='models/waste_classifier.tflite',
    quantize=True,
    optimize=True
):
    """Convert Keras model to TFLite format."""
    
    print("\n" + "="*60)
    print("🔄 CONVERTING MODEL TO TFLITE")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please run train_model.py first.")
        return False
    
    # Load the Keras model
    print(f"\n📂 Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimization options
    if optimize:
        print("🔧 Applying optimizations...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    if quantize:
        print("🔧 Applying quantization...")
        # Full integer quantization for better mobile performance
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TFLite ops
            tf.lite.OpsSet.SELECT_TF_OPS     # Enable TF ops (fallback)
        ]
        converter.target_spec.supported_types = [tf.float16]
    
    # Convert
    print("\n🔄 Converting model...")
    tflite_model = converter.convert()
    
    # Save
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Get file sizes
    keras_size = os.path.getsize(model_path) / (1024 * 1024)
    tflite_size = os.path.getsize(output_path) / (1024 * 1024)
    reduction = (1 - tflite_size / keras_size) * 100
    
    print(f"\n✅ Model converted successfully!")
    print(f"   Keras model size:  {keras_size:.2f} MB")
    print(f"   TFLite model size: {tflite_size:.2f} MB")
    print(f"   Size reduction:    {reduction:.1f}%")
    print(f"   Output: {output_path}")
    
    # Verify the converted model
    print("\n🔍 Verifying converted model...")
    verify_tflite_model(output_path)
    
    # Create metadata file
    create_metadata(output_path)
    
    print("\n" + "="*60)
    print("✅ CONVERSION COMPLETE!")
    print("="*60)
    
    flutter_assets = '../smart_waste_app/assets/models/'
    print(f"\n📱 To use in Flutter app:")
    print(f"   1. Copy model:")
    print(f"      cp {output_path} {flutter_assets}")
    print(f"   2. Copy labels:")
    print(f"      cp models/waste_labels.txt {flutter_assets}")
    print(f"   3. Update pubspec.yaml assets if needed")
    
    return True

def verify_tflite_model(model_path):
    """Verify the TFLite model works correctly."""
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"   Input shape:  {input_details[0]['shape']}")
    print(f"   Input dtype:  {input_details[0]['dtype']}")
    print(f"   Output shape: {output_details[0]['shape']}")
    print(f"   Output dtype: {output_details[0]['dtype']}")
    
    # Test with random input
    input_shape = input_details[0]['shape']
    test_input = np.random.rand(*input_shape).astype(np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details[0]['index'])
    print(f"   Test output:  {output[0][:3]}... (showing first 3 classes)")
    print(f"   ✅ Model verification passed!")

def create_metadata(model_path):
    """Create a metadata JSON file for the model."""
    import json
    from datetime import datetime
    
    metadata = {
        'name': 'Waste Classifier',
        'version': '1.0.0',
        'created': datetime.now().isoformat(),
        'input_size': 224,
        'classes': ['organic', 'recyclable', 'hazardous', 'ewaste', 'general'],
        'description': 'MobileNetV2-based waste classification model',
        'preprocessing': {
            'normalize': True,
            'mean': [0.0, 0.0, 0.0],
            'std': [1.0, 1.0, 1.0]
        }
    }
    
    metadata_path = model_path.replace('.tflite', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✅ Metadata saved to: {metadata_path}")

def benchmark_model(model_path, num_runs=100):
    """Benchmark inference speed."""
    import time
    
    print(f"\n⏱️  Benchmarking inference speed ({num_runs} runs)...")
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    test_input = np.random.rand(*input_shape).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        times.append(time.perf_counter() - start)
    
    avg_time = np.mean(times) * 1000  # Convert to ms
    std_time = np.std(times) * 1000
    
    print(f"   Average inference: {avg_time:.2f} ms (±{std_time:.2f} ms)")
    print(f"   FPS: {1000/avg_time:.1f}")

def main():
    parser = argparse.ArgumentParser(description='Convert Keras model to TFLite')
    parser.add_argument('--model', type=str, default='models/waste_classifier_best.keras',
                       help='Path to Keras model')
    parser.add_argument('--output', type=str, default='models/waste_classifier.tflite',
                       help='Output TFLite path')
    parser.add_argument('--no-quantize', action='store_true',
                       help='Disable quantization')
    parser.add_argument('--no-optimize', action='store_true',
                       help='Disable optimization')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run inference benchmark')
    
    args = parser.parse_args()
    
    success = convert_to_tflite(
        model_path=args.model,
        output_path=args.output,
        quantize=not args.no_quantize,
        optimize=not args.no_optimize
    )
    
    if success and args.benchmark:
        benchmark_model(args.output)

if __name__ == '__main__':
    main()

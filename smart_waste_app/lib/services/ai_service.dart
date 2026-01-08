import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:flutter/foundation.dart';

/// AI Service for Smart Waste App
/// Provides:
/// 1. Waste Image Classification (using Google Cloud Vision or local rules)
/// 2. ChatBot AI responses (using OpenAI GPT)
class AIService {
  // TODO: Replace with your actual API key
  // Get your API key from: https://platform.openai.com/api-keys
  static const String _openAIApiKey = 'YOUR_OPENAI_API_KEY';
  
  // OpenAI API endpoint
  static const String _openAIEndpoint = 'https://api.openai.com/v1/chat/completions';

  /// Check if OpenAI is configured
  static bool get isOpenAIConfigured => _openAIApiKey != 'YOUR_OPENAI_API_KEY' && _openAIApiKey.isNotEmpty;

  /// Get AI response from OpenAI GPT
  /// Falls back to rule-based responses if API key is not configured
  static Future<String> getChatResponse(String userMessage) async {
    if (!isOpenAIConfigured) {
      return _getRuleBasedResponse(userMessage);
    }

    try {
      final response = await http.post(
        Uri.parse(_openAIEndpoint),
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer $_openAIApiKey',
        },
        body: jsonEncode({
          'model': 'gpt-3.5-turbo',
          'messages': [
            {
              'role': 'system',
              'content': '''You are EcoBot, a helpful AI assistant for the Smart Waste Collection app. 
Your role is to help users with:
- Scheduling waste pickups
- Understanding waste types (Organic, Recyclable, E-Waste, Hazardous, General)
- Eco Points and rewards system
- Tracking pickup status
- General waste management and recycling tips

Be friendly, concise, and use emojis occasionally. Always encourage eco-friendly practices.
Keep responses under 150 words unless detailed explanation is needed.

Available waste types:
- Organic: Food scraps, garden waste, biodegradables
- Recyclable: Paper, plastic, glass, metal  
- E-Waste: Electronics, batteries, cables
- Hazardous: Chemicals, paint, medical waste
- General: Mixed non-recyclable waste

Eco Points: Users earn 25 points per completed pickup, bonus 10 points every 5 pickups.
Rewards: 100 pts = ₹50 voucher, 250 pts = ₹150 coupon, 500 pts = ₹300 essentials, 1000 pts = ₹500 + free month'''
            },
            {
              'role': 'user',
              'content': userMessage,
            }
          ],
          'max_tokens': 300,
          'temperature': 0.7,
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return data['choices'][0]['message']['content'].toString().trim();
      } else {
        if (kDebugMode) {
          print('OpenAI API Error: ${response.statusCode} - ${response.body}');
        }
        return _getRuleBasedResponse(userMessage);
      }
    } catch (e) {
      if (kDebugMode) {
        print('AI Service Error: $e');
      }
      return _getRuleBasedResponse(userMessage);
    }
  }

  /// Classify waste from image using AI
  /// Returns the predicted waste type and confidence
  static Future<WasteClassificationResult> classifyWasteFromImage(String base64Image) async {
    // For now, we'll use a simple keyword-based classification
    // In production, integrate with:
    // - TensorFlow Lite model
    // - Google Cloud Vision API
    // - Custom ML model
    
    // Simulated AI classification with random but realistic results
    // In real implementation, this would analyze the actual image
    
    await Future.delayed(const Duration(seconds: 2)); // Simulate processing
    
    // Return a default result - in production this would be actual ML inference
    return WasteClassificationResult(
      wasteType: 'Recyclable',
      confidence: 0.85,
      suggestions: [
        'This appears to be recyclable material',
        'Please ensure the item is clean and dry',
        'Remove any food residue before disposal',
      ],
      disposalTips: 'Place in the blue recycling bin. If contaminated with food, wash before recycling.',
    );
  }

  /// Analyze waste image and detect objects
  /// Uses Google Cloud Vision API or local ML model
  static Future<List<DetectedWasteItem>> detectWasteItems(String base64Image) async {
    // Simulated detection - in production use actual ML
    await Future.delayed(const Duration(seconds: 1));
    
    return [
      DetectedWasteItem(
        name: 'Plastic Bottle',
        category: 'Recyclable',
        confidence: 0.92,
        recyclable: true,
      ),
    ];
  }

  /// Rule-based response fallback when OpenAI is not available
  static String _getRuleBasedResponse(String userMessage) {
    final message = userMessage.toLowerCase().trim();

    // Greetings
    if (message.contains('hello') || message.contains('hi') || message.contains('hey')) {
      return "Hello! 😊 I'm EcoBot, powered by AI! How can I assist you today with your waste management needs?";
    }

    // Pickup scheduling
    if (message.contains('schedule') || message.contains('book') || message.contains('new pickup')) {
      return "To schedule a new pickup:\n\n"
          "1️⃣ Go to Home screen\n"
          "2️⃣ Tap the '+ New Pickup' button\n"
          "3️⃣ Select your waste type (or use AI Scan! 📸)\n"
          "4️⃣ Choose quantity and time slot\n"
          "5️⃣ Confirm your address\n"
          "6️⃣ Submit your request!\n\n"
          "💡 Tip: Use the AI camera to automatically detect waste type!";
    }

    // Track pickup
    if (message.contains('track') || message.contains('status') || message.contains('where is')) {
      return "To track your pickup:\n\n"
          "📍 Go to Home → 'Your Pickups' section\n"
          "📱 Tap on any pickup to see real-time status\n\n"
          "Status meanings:\n"
          "🟡 Pending - Waiting for collector\n"
          "🔵 Assigned - Collector assigned\n"
          "🟠 In Progress - On the way!\n"
          "🟢 Completed - Done!";
    }

    // Eco points / rewards
    if (message.contains('eco') || message.contains('point') || message.contains('reward')) {
      return "🌱 Eco Points System:\n\n"
          "• +25 points per completed pickup\n"
          "• +10 bonus every 5 pickups\n\n"
          "🎁 Redeem Rewards:\n"
          "• 100 pts → ₹50 Grocery Voucher\n"
          "• 250 pts → ₹150 Shopping Coupon\n"
          "• 500 pts → ₹300 Home Essentials\n"
          "• 1000 pts → ₹500 + Free Month!\n\n"
          "♻️ Recycle more, earn more!";
    }

    // Waste types
    if (message.contains('waste type') || message.contains('recycle') || message.contains('what type') || message.contains('classify')) {
      return "♻️ Waste Categories:\n\n"
          "🌱 Organic - Food, garden waste\n"
          "♻️ Recyclable - Paper, plastic, glass, metal\n"
          "💻 E-Waste - Electronics, batteries\n"
          "⚠️ Hazardous - Chemicals, medical\n"
          "🗑️ General - Mixed waste\n\n"
          "📸 Pro Tip: Use our AI Scanner to auto-detect waste type from a photo!";
    }

    // AI/scan features
    if (message.contains('scan') || message.contains('camera') || message.contains('photo') || message.contains('ai') || message.contains('detect')) {
      return "📸 AI Waste Scanner:\n\n"
          "Our smart camera can identify waste types automatically!\n\n"
          "How to use:\n"
          "1️⃣ Go to 'New Pickup'\n"
          "2️⃣ Tap '📸 AI Scan' button\n"
          "3️⃣ Take a photo of your waste\n"
          "4️⃣ AI will detect and classify it!\n\n"
          "🤖 Powered by machine learning for accurate classification.";
    }

    // Default
    return "I'm here to help! 🌍\n\n"
        "You can ask me about:\n"
        "• 📅 Scheduling pickups\n"
        "• 📍 Tracking your requests\n"
        "• 🏆 Eco points & rewards\n"
        "• ♻️ Waste types & recycling\n"
        "• 📸 AI waste scanning\n\n"
        "What would you like to know?";
  }
}

/// Result of waste classification
class WasteClassificationResult {
  final String wasteType;
  final double confidence;
  final List<String> suggestions;
  final String disposalTips;

  WasteClassificationResult({
    required this.wasteType,
    required this.confidence,
    required this.suggestions,
    required this.disposalTips,
  });

  String get confidencePercentage => '${(confidence * 100).toStringAsFixed(0)}%';
  
  bool get isHighConfidence => confidence >= 0.8;
}

/// Detected waste item from image
class DetectedWasteItem {
  final String name;
  final String category;
  final double confidence;
  final bool recyclable;

  DetectedWasteItem({
    required this.name,
    required this.category,
    required this.confidence,
    required this.recyclable,
  });
}

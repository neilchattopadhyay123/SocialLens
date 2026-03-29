#include "EmotionDetector.hpp"

int main() {
    // 1. Setup the Detector
    EmotionDetector detector("emo_affectnet.onnx");
    
    // 2. In your SmartSpectra Video Callback
    sdk.setOnVideoFrame([&](cv::Mat frame, FaceData face) {
        
        // CROP: Use the SDK's face coordinates
        cv::Rect roi(face.x, face.y, face.width, face.height);
        cv::Mat face_crop = frame(roi);

        // PREDICT: Get the emotion word
        std::string emotion = detector.predict(face_crop);

        // DISPLAY: Draw it on the video
        cv::putText(frame, "Emotion: " + emotion, cv::Point(30, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        cv::imshow("SocialLens Monitor", frame);
        cv::waitKey(1);
    });
}
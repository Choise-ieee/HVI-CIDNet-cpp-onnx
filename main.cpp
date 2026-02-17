#include <windows.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// ==================== Utility Functions ====================

std::wstring s2ws(const std::string& str) {
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, nullptr, 0);
    std::wstring wstr(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, &wstr[0], size_needed);
    return wstr;
}

// ==================== Image Processing ====================

// Normalize + BGR to RGB + Resize + HWC to CHW
std::vector<float> preprocess(const cv::Mat& img, int target_w, int target_h) {
    // BGR -> RGB
    cv::Mat img_rgb;
    cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);

    // Resize to target size
    cv::Mat img_resized;
    cv::resize(img_rgb, img_resized, cv::Size(target_w, target_h), 0, 0, cv::INTER_LINEAR);

    // Normalize to [0, 1]
    cv::Mat img_float;
    img_resized.convertTo(img_float, CV_32FC3, 1.0 / 255.0);

    // HWC -> CHW
    std::vector<float> tensor(3 * target_h * target_w);
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < target_h; ++h) {
            for (int w = 0; w < target_w; ++w) {
                tensor[c * target_h * target_w + h * target_w + w] =
                    img_float.ptr<float>(h)[w * 3 + c];
            }
        }
    }
    return tensor;
}

// Apply Gamma Correction
void applyGamma(std::vector<float>& tensor, float gamma) {
    if (gamma == 1.0f) return;
    for (auto& v : tensor) {
        v = powf(std::max(0.0f, std::min(1.0f, v)), gamma);
    }
}

// CHW -> HWC
cv::Mat nchwToHwc(const std::vector<float>& tensor, int H, int W) {
    cv::Mat img(H, W, CV_32FC3);
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            for (int c = 0; c < 3; ++c) {
                img.ptr<float>(h)[w * 3 + c] = tensor[c * H * W + h * W + w];
            }
        }
    }
    return img;
}

// ==================== Main Function ====================

int main(int argc, char** argv) {
    // Parameters
    std::string onnx_path = "HVI-CIDNet-Generalization.onnx";
    std::string image_path = "6.png";
    std::string output_path = "result.png";
    float alpha_s = 1.0f, alpha_i = 1.0f, gamma = 1.0f;
    bool use_cuda = true;  // Enable CUDA by default

    // Model input size (640x480)
    int input_width = 640;
    int input_height = 480;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--onnx" && i + 1 < argc) onnx_path = argv[++i];
        else if (arg == "--input" && i + 1 < argc) image_path = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
        else if (arg == "--alpha_s" && i + 1 < argc) alpha_s = std::stof(argv[++i]);
        else if (arg == "--alpha_i" && i + 1 < argc) alpha_i = std::stof(argv[++i]);
        else if (arg == "--gamma" && i + 1 < argc) gamma = std::stof(argv[++i]);
        else if (arg == "--width" && i + 1 < argc) input_width = std::stoi(argv[++i]);
        else if (arg == "--height" && i + 1 < argc) input_height = std::stoi(argv[++i]);
        else if (arg == "--cpu") use_cuda = false;
        else if (arg[0] != '-') {
            if (onnx_path == "HVI-CIDNet-Generalization.onnx") onnx_path = arg;
            else if (image_path == "input.png") image_path = arg;
        }
    }

    std::cout << "=== HVI-CIDNet Inference ===" << std::endl;
    std::cout << "Model: " << onnx_path << std::endl;
    std::cout << "Input: " << image_path << std::endl;
    std::cout << "Output: " << output_path << std::endl;
    std::cout << "Input Size: " << input_width << "x" << input_height << std::endl;
    std::cout << "Params: alpha_s=" << alpha_s << ", alpha_i=" << alpha_i << ", gamma=" << gamma << std::endl;
    std::cout << "Device: " << (use_cuda ? "CUDA" : "CPU") << std::endl;

    // ========== 1. Load Image ==========
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: Failed to load image: " << image_path << std::endl;
        return -1;
    }
    int orig_h = img.rows;
    int orig_w = img.cols;
    std::cout << "Original image: " << orig_w << "x" << orig_h << std::endl;

    // ========== 2. Preprocess (Resize + RGB + Normalize + HWC->CHW) ==========
    std::vector<float> input_tensor = preprocess(img, input_width, input_height);
    std::cout << "Resized to: " << input_width << "x" << input_height << std::endl;

    // ========== 3. Apply Gamma ==========
    applyGamma(input_tensor, gamma);

    // ========== 4. ONNX Inference ==========
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "HVI_CIDNet");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // Enable CUDA Execution Provider
    if (use_cuda) {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;  // GPU 0
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;

        try {
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "CUDA execution provider enabled successfully." << std::endl;
        }
        catch (const Ort::Exception& e) {
            std::cerr << "Warning: Failed to enable CUDA: " << e.what() << std::endl;
            std::cerr << "Falling back to CPU execution." << std::endl;
            use_cuda = false;
        }
    }

    std::wstring onnx_path_w = s2ws(onnx_path);
    Ort::Session session(env, onnx_path_w.c_str(), session_options);

    // Input tensors
    std::vector<const char*> input_names = { "input", "alpha_s", "alpha_i", "gamma" };
    std::vector<int64_t> input_shape = { 1, 3, input_height, input_width };
    std::vector<int64_t> scalar_shape = { 1 };

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
        input_tensor.data(), input_tensor.size(),
        input_shape.data(), input_shape.size()
        ));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
        &alpha_s, 1, scalar_shape.data(), scalar_shape.size()
        ));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
        &alpha_i, 1, scalar_shape.data(), scalar_shape.size()
        ));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
        &gamma, 1, scalar_shape.data(), scalar_shape.size()
        ));

    // Inference
    std::vector<const char*> output_names = { "output" };
    auto output_tensors = session.Run(
        Ort::RunOptions{ nullptr },
        input_names.data(), input_tensors.data(), input_names.size(),
        output_names.data(), 1
    );

    // ========== 5. Postprocess ==========
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto info = output_tensors[0].GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();
    int out_H = static_cast<int>(shape[2]);
    int out_W = static_cast<int>(shape[3]);
    size_t output_size = info.GetElementCount();

    std::cout << "Output size: " << out_W << "x" << out_H << std::endl;

    std::vector<float> output_tensor(output_data, output_data + output_size);

    // Clamp [0, 1]
    for (auto& v : output_tensor) {
        v = std::max(0.0f, std::min(1.0f, v));
    }

    // CHW -> HWC
    cv::Mat output_hwc = nchwToHwc(output_tensor, out_H, out_W);

    // Resize back to original size
    cv::Mat output_resized;
    cv::resize(output_hwc, output_resized, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_LINEAR);

    // Save (HWC RGB -> BGR)
    cv::Mat output_bgr;
    output_resized.convertTo(output_bgr, CV_8UC3, 255.0);
    cv::cvtColor(output_bgr, output_bgr, cv::COLOR_RGB2BGR);

    cv::imwrite("results.jpg", output_bgr);
    std::cout << "Done! Result saved to: " << output_path << std::endl;
    Sleep(99999);
    return 0;
}

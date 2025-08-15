// nanodet_core.h
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

struct Object {
    cv::Rect_<float> rect;
    int   label;
    float prob;
};

void nanodet_init(const char* param_path, const char* bin_path, int threads = 4);
int  detect_nanodet(const cv::Mat& bgr, std::vector<Object>& objects);

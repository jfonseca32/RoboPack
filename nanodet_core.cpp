// nanodet_core.cpp
#include <ncnn/net.h>        // <-- use namespaced includes
#include <ncnn/layer.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cfloat>
#include <vector>
#include "nanodet_core.h"

// ---- model + params (same as your file) ----
static ncnn::Net g_nanodet;

static const int   target_size    = 320;
static const float prob_threshold = 0.4f;
static const float nms_threshold  = 0.5f;
static const float mean_vals[3]   = {103.53f, 116.28f, 123.675f};
static const float norm_vals[3]   = {0.017429f, 0.017507f, 0.017125f};

// ---- helpers copied from your nanodet.cpp ----
static inline float intersection_area(const Object& a, const Object& b) {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& objs, int left, int right) {
    int i = left, j = right;
    float p = objs[(left + right) / 2].prob;
    while (i <= j) {
        while (objs[i].prob > p) ++i;
        while (objs[j].prob < p) --j;
        if (i <= j) { std::swap(objs[i], objs[j]); ++i; --j; }
    }
#pragma omp parallel sections
    {
#pragma omp section
        { if (left < j) qsort_descent_inplace(objs, left, j); }
#pragma omp section
        { if (i < right) qsort_descent_inplace(objs, i, right); }
    }
}
static void qsort_descent_inplace(std::vector<Object>& objs) {
    if (!objs.empty()) qsort_descent_inplace(objs, 0, (int)objs.size()-1);
}

static void nms_sorted_bboxes(const std::vector<Object>& objs, std::vector<int>& picked, float nms_thr) {
    picked.clear();
    const int n = (int)objs.size();
    std::vector<float> areas(n);
    for (int i=0;i<n;i++) areas[i] = objs[i].rect.area();
    for (int i=0;i<n;i++) {
        const Object& a = objs[i];
        bool keep = true;
        for (int j : picked) {
            const Object& b = objs[j];
            float inter = intersection_area(a,b);
            float uni   = areas[i] + areas[j] - inter;
            if (inter / uni > nms_thr) { keep = false; break; }
        }
        if (keep) picked.push_back(i);
    }
}

static void generate_proposals(const ncnn::Mat& cls_pred, const ncnn::Mat& dis_pred,
                               int stride, const ncnn::Mat& in_pad,
                               float prob_thr, std::vector<Object>& objects)
{
    const int num_grid = cls_pred.h;
    int num_grid_x, num_grid_y;
    if (in_pad.w > in_pad.h) {
        num_grid_x = in_pad.w / stride;
        num_grid_y = num_grid / num_grid_x;
    } else {
        num_grid_y = in_pad.h / stride;
        num_grid_x = num_grid / num_grid_y;
    }

    const int num_class = cls_pred.w;
    const int reg_max_1 = dis_pred.w / 4;

    for (int i = 0; i < num_grid_y; i++) {
        for (int j = 0; j < num_grid_x; j++) {
            const int idx = i * num_grid_x + j;
            const float* scores = cls_pred.row(idx);

            int label = -1; float score = -FLT_MAX;
            for (int k=0;k<num_class;k++) if (scores[k] > score) { label=k; score=scores[k]; }
            if (score < prob_thr) continue;

            ncnn::Mat bbox_pred(reg_max_1, 4, (void*)dis_pred.row(idx));
            {   // softmax along reg_max dim
                ncnn::Layer* softmax = ncnn::create_layer("Softmax");
                ncnn::ParamDict pd; pd.set(0, 1); pd.set(1, 1); // axis=1
                softmax->load_param(pd);
                ncnn::Option opt; opt.num_threads=1; opt.use_packing_layout=false;
                softmax->create_pipeline(opt);
                softmax->forward_inplace(bbox_pred, opt);
                softmax->destroy_pipeline(opt);
                delete softmax;
            }
            float pred_ltrb[4];
            for (int k=0;k<4;k++) {
                float dis = 0.f;
                const float* p = bbox_pred.row(k);
                for (int l=0;l<reg_max_1;l++) dis += l * p[l];
                pred_ltrb[k] = dis * stride;
            }
            float pb_cx = (j + 0.5f) * stride;
            float pb_cy = (i + 0.5f) * stride;

            Object obj;
            obj.rect.x = pb_cx - pred_ltrb[0];
            obj.rect.y = pb_cy - pred_ltrb[1];
            obj.rect.width  = pred_ltrb[0] + pred_ltrb[2];
            obj.rect.height = pred_ltrb[1] + pred_ltrb[3];
            obj.label = label;
            obj.prob  = score;
            objects.push_back(obj);
        }
    }
}

// ---- public API ----
void nanodet_init(const char* param_path, const char* bin_path, int threads) {
    g_nanodet.load_param(param_path);
    g_nanodet.load_model(bin_path);
    g_nanodet.opt.num_threads = threads;
}

int detect_nanodet(const cv::Mat& bgr, std::vector<Object>& objects)
{
    const int width = bgr.cols;
    const int height = bgr.rows;

    // resize with aspect
    int w = width, h = height; float scale = 1.f;
    if (w > h) { scale = (float)target_size / w; w = target_size; h = int(h * scale); }
    else       { scale = (float)target_size / h; h = target_size; w = int(w * scale); }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, width, height, w, h);

    // pad to multiple of 32
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad/2, hpad - hpad/2, wpad/2, wpad - wpad/2, ncnn::BORDER_CONSTANT, 0.f);

    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = g_nanodet.create_extractor();
    ex.input("input.1", in_pad);

    std::vector<Object> proposals;

    // stride 8
    {
        ncnn::Mat cls_pred, dis_pred;
        ex.extract("792", cls_pred);
        ex.extract("795", dis_pred);
        std::vector<Object> tmp; generate_proposals(cls_pred, dis_pred, 8, in_pad, prob_threshold, tmp);
        proposals.insert(proposals.end(), tmp.begin(), tmp.end());
    }
    // stride 16
    {
        ncnn::Mat cls_pred, dis_pred;
        ex.extract("814", cls_pred);
        ex.extract("817", dis_pred);
        std::vector<Object> tmp; generate_proposals(cls_pred, dis_pred, 16, in_pad, prob_threshold, tmp);
        proposals.insert(proposals.end(), tmp.begin(), tmp.end());
    }
    // stride 32
    {
        ncnn::Mat cls_pred, dis_pred;
        ex.extract("836", cls_pred);
        ex.extract("839", dis_pred);
        std::vector<Object> tmp; generate_proposals(cls_pred, dis_pred, 32, in_pad, prob_threshold, tmp);
        proposals.insert(proposals.end(), tmp.begin(), tmp.end());
    }

    qsort_descent_inplace(proposals);
    std::vector<int> picked; nms_sorted_bboxes(proposals, picked, nms_threshold);

    objects.resize((int)picked.size());
    // map back to original image space
    for (int pi=0; pi<(int)picked.size(); ++pi) {
        Object o = proposals[picked[pi]];
        float x0 = (o.rect.x                      - (wpad/2)) / scale;
        float y0 = (o.rect.y                      - (hpad/2)) / scale;
        float x1 = (o.rect.x + o.rect.width  - (wpad/2)) / scale;
        float y1 = (o.rect.y + o.rect.height - (hpad/2)) / scale;
        x0 = std::max(std::min(x0, (float)(width  - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(width  - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);
        objects[pi].rect   = cv::Rect_<float>(x0,y0,x1-x0,y1-y0);
        objects[pi].label  = o.label;
        objects[pi].prob   = o.prob;
    }
    return 0;
}

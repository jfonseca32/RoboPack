// hybrid_follow_avoid.cpp  — smoothed blobs, person-ROI exclusion, dynamic cadence
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include "nanodet_core.h"

static void split_res(const std::string& s, int& w, int& h) {
    auto x = s.find('x'); w = 640; h = 480;
    if (x != std::string::npos) { w = std::stoi(s.substr(0,x)); h = std::stoi(s.substr(x+1)); }
}
static void usage() {
    std::cerr << "Usage: ./hybrid_app --source usb0|picamera0 [--resolution 640x480] [--det_every 8]\n";
}

int main(int argc, char** argv) {
    std::string source, res="640x480"; int det_every_flag=8;
    for (int i=1;i<argc;++i){
        std::string a=argv[i]; auto next=[&](std::string& d){ if(i+1<argc) d=argv[++i]; };
        if(a=="--source") next(source);
        else if(a=="--resolution") next(res);
        else if(a=="--det_every"){ std::string t; next(t); det_every_flag=std::max(1,std::stoi(t)); }
        else { std::cerr<<"Unknown arg: "<<a<<"\n"; usage(); return 1; }
    }
    if(source.empty()){ usage(); return 1; }
    int W,H; split_res(res,W,H);

    // camera
    cv::VideoCapture cap;
    if(source.rfind("usb",0)==0){ int idx=std::stoi(source.substr(3)); cap.open(idx); }
    else if(source.rfind("picamera",0)==0){ cap.open(0); } // if PiCam is not /dev/video*, use libcamera/gstreamer pipeline
    else { std::cerr<<"Invalid --source\n"; return 1; }
    if(!cap.isOpened()){ std::cerr<<"Camera open failed\n"; return 1; }
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  W);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, H);

    // ---- Blobs (MOG2 background subtractor) ----
    // detectShadows=true gives 127 for shadows. We'll strip them out.
    cv::Ptr<cv::BackgroundSubtractorMOG2> bg = cv::createBackgroundSubtractorMOG2(500, 16.0, true);
    float bg_learn_rate = 0.005f;   // slow learning => less flicker
    // Temporal smoothing accumulator (float 0..1)
    cv::Mat fg_acc;                 // CV_32F
    float acc_alpha = 0.1f;         // higher = faster response, lower = smoother

    // ROI focusing on lower part of frame (e.g., floor-level obstacles)
    cv::Rect roi(0, (int)(H*0.35), W, H - (int)(H*0.35)); // bottom 65%
    int min_area = std::max(600, (W*H)/600);  // scale with resolution

    // ---- NanoDet init ----
    nanodet_init("nanodet_m.param", "nanodet_m.bin", 4);
    const int PERSON_CLASS_ID = 0;

    bool have_person=false;
    cv::Rect person_box;   // int rect for drawing/UI

    // dynamic cadence: quicker reacquire when lost
    int det_every_hit  = det_every_flag; // e.g., 8
    int det_every_miss = std::max(2, det_every_flag/2); // e.g., 4 -> 2, 8 -> 4

    std::vector<double> fps_hist; fps_hist.reserve(200);
    int frame_idx=0;

    for(;;){
        auto t0 = std::chrono::high_resolution_clock::now();

        cv::Mat frame;
        if(!cap.read(frame) || frame.empty()) break;
        cv::resize(frame, frame, cv::Size(W,H));

        // ---------- BLOB STAGE ----------
        cv::Mat fg_raw;
        bg->apply(frame, fg_raw, bg_learn_rate);

        // keep only 255 (sure foreground), drop shadows (127)
        cv::Mat fg;
        cv::threshold(fg_raw, fg, 254, 255, cv::THRESH_BINARY);

        // apply ROI (zero outside)
        {
            cv::Mat mask = cv::Mat::zeros(fg.size(), fg.type());
            mask(roi).setTo(255);
            cv::bitwise_and(fg, mask, fg);
        }

        // ignore person region (no avoidance against our target)
        if (have_person) {
            cv::Rect clampPB = person_box & cv::Rect(0,0,frame.cols,frame.rows);
            if (clampPB.area() > 0) cv::rectangle(fg, clampPB, 0, cv::FILLED);
        }

        // temporal smoothing via running average
        cv::Mat fg_float;
        fg.convertTo(fg_float, CV_32F, 1.0/255.0);
        if (fg_acc.empty()) {
            fg_float.copyTo(fg_acc);
        } else {
            cv::accumulateWeighted(fg_float, fg_acc, acc_alpha);
        }
        cv::Mat fg_stable;
        cv::threshold(fg_acc, fg_stable, 0.5, 1.0, cv::THRESH_BINARY); // majority-ish
        fg_stable.convertTo(fg, CV_8U, 255.0);

        // morphology to stabilize shapes
        cv::morphologyEx(fg, fg, cv::MORPH_OPEN,  cv::getStructuringElement(cv::MORPH_ELLIPSE,{3,3}));
        cv::morphologyEx(fg, fg, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE,{9,9}));

        // contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(fg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for(const auto& c: contours){
            double area=cv::contourArea(c);
            if(area<min_area) continue;
            cv::Rect r=cv::boundingRect(c);
            cv::rectangle(frame, r, {0,255,0}, 2); // green = obstacle
        }

        // ---------- PERSON DETECTION ----------
        int cadence = have_person ? det_every_hit : det_every_miss;
        if(!have_person || (frame_idx % cadence)==0){
            std::vector<Object> objects;
            detect_nanodet(frame, objects);

            int best=-1; float bestScore=-1.f; double bestArea=-1.0;
            for(int i=0;i<(int)objects.size();++i){
                if(objects[i].label!=PERSON_CLASS_ID) continue;
                double area=objects[i].rect.area();
                if(objects[i].prob>bestScore || (objects[i].prob==bestScore && area>bestArea)){
                    bestScore=objects[i].prob; bestArea=area; best=i;
                }
            }
            if(best>=0){
                cv::Rect pb = objects[best].rect;                    // float->int cast OK
                cv::Rect frameRect(0,0,frame.cols,frame.rows);
                person_box = pb & frameRect;
                have_person = (person_box.area() > 0);
            } else {
                have_person = false;
            }
        }

        // draw person
        if(have_person){
            cv::rectangle(frame, person_box, {0,0,255}, 2);
            cv::Point label_pt(person_box.x, std::max(0, person_box.y - 5));
            cv::putText(frame, "person", label_pt, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,0,255}, 2);
        }

        // ---------- FPS ----------
        auto t1 = std::chrono::high_resolution_clock::now();
        double fps = 1e9 / std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        if((int)fps_hist.size()>=200) fps_hist.erase(fps_hist.begin());
        fps_hist.push_back(fps);
        double sum=0; for(double x: fps_hist) sum+=x; double fps_avg=sum/fps_hist.size();
        cv::putText(frame, ("FPS: "+std::to_string(fps_avg)).substr(0,12),
                    {10,20}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,255,255}, 2);

        cv::imshow("Hybrid (Blobs + Person)", frame);
        int key=cv::waitKey(1);
        if((key&255)=='q') break;
        if((key&255)=='s') cv::waitKey();
        frame_idx++;
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}

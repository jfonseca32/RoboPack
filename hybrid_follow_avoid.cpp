// hybrid_follow_avoid.cpp  — no tracker dependency (works without opencv_contrib)
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
    std::string source, res="640x480"; int det_every=8;
    for (int i=1;i<argc;++i){
        std::string a=argv[i]; auto next=[&](std::string& d){ if(i+1<argc) d=argv[++i]; };
        if(a=="--source") next(source);
        else if(a=="--resolution") next(res);
        else if(a=="--det_every"){ std::string t; next(t); det_every=std::max(1,std::stoi(t)); }
        else { std::cerr<<"Unknown arg: "<<a<<"\n"; usage(); return 1; }
    }
    if(source.empty()){ usage(); return 1; }
    int W,H; split_res(res,W,H);

    // camera
    cv::VideoCapture cap;
    if(source.rfind("usb",0)==0){ int idx=std::stoi(source.substr(3)); cap.open(idx); }
    else if(source.rfind("picamera",0)==0){ cap.open(0); } // if PiCam not at /dev/video*, use a libcamera/gstreamer pipeline
    else { std::cerr<<"Invalid --source\n"; return 1; }
    if(!cap.isOpened()){ std::cerr<<"Camera open failed\n"; return 1; }
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  W);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, H);

    // blobs
    cv::Ptr<cv::BackgroundSubtractor> bg = cv::createBackgroundSubtractorMOG2(300,16.0,true);

    // NanoDet init (model input stays 320x320 internally)
    nanodet_init("nanodet_m.param", "nanodet_m.bin", 4);
    const int PERSON_CLASS_ID = 0;

    bool have_person=false;
    cv::Rect person_box;   // int rect for drawing/UI
    std::vector<double> fps_hist; fps_hist.reserve(200);
    int frame_idx=0;

    for(;;){
        auto t0 = std::chrono::high_resolution_clock::now();

        cv::Mat frame;
        if(!cap.read(frame) || frame.empty()) break;
        cv::resize(frame, frame, cv::Size(W,H));

        // --- Blobs (class-agnostic) ---
        cv::Mat fg;
        bg->apply(frame, fg);
        cv::erode(fg, fg, cv::getStructuringElement(cv::MORPH_ELLIPSE,{3,3}));
        cv::dilate(fg, fg, cv::getStructuringElement(cv::MORPH_ELLIPSE,{7,7}), cv::Point(-1,-1), 2);
        cv::threshold(fg, fg, 200, 255, cv::THRESH_BINARY);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(fg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for(const auto& c: contours){
            double area=cv::contourArea(c);
            if(area<800) continue;
            cv::Rect r=cv::boundingRect(c);
            cv::rectangle(frame, r, {0,255,0}, 2); // green = obstacle
        }

        // --- Person detection every N frames; reuse last box in between ---
        if(!have_person || (frame_idx % det_every)==0){
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
                // convert Rect_<float> -> Rect (int) and clamp to frame
                cv::Rect pb = objects[best].rect;
                cv::Rect frameRect(0,0,frame.cols,frame.rows);
                person_box = pb & frameRect;
                have_person = (person_box.area() > 0);
            } else {
                have_person = false;
            }
        }
        // else: keep last person_box as-is for this frame (no tracker)

        // --- Draw person box ---
        if(have_person){
            cv::rectangle(frame, person_box, {0,0,255}, 2);
            cv::Point label_pt(person_box.x, std::max(0, person_box.y - 5));
            cv::putText(frame, "person", label_pt, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,0,255}, 2);
        }

        // --- FPS overlay ---
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

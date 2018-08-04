// daniel last email start

// Detection of circular bounding box
// Hasnat ASE Master Thesis
#include "opencv2/imgcodecs.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include <cv.h>

#include <iostream>
#include <vector>



using namespace cv;
using namespace std;



struct recircle{
        float x;
        float y;
        float r;
        float redetects;
        int typ;

        recircle(float x, float y, float r){
                this->x = x; this->y = y; this->r = r; this->redetects = 0; typ=0;
        }
};

Mat image, imagegray, canny_output;
int rmin = 35;
int rmax = 35;
int cannythreshmax = 110;
int centerthresh = 11;
int cthreshl = 44;//10;
int cthreshh = 138;
int maxapproxdist = 20;
int minbbheight = 123;
int minbbwidth = 21;
int minbbheight2 = 45;
int minbbwidth2 = 8;
vector<recircle> circlespm;
int maxhredetects = 0;
int maxhredetects3 = 0;
size_t maxhidx = 0;

float seg12 = 0;
float seg23 = 0;
float seg13 = 0;

//für alle circle
//search in model circle list
//if close h++
//else insert mit h=1
//h-- for all

int dist(Point p1, Point p2){

        int dx = p1.x-p2.x;
        int dy = p1.y-p2.y;

        return sqrt(dx*dx+dy*dy);


}

void writeText(std::ostringstream& oss, int x, int y){


        putText(image,
          oss.str(),
          Point(x, y),
          FONT_HERSHEY_SIMPLEX,
          0.6,
          Scalar(0,255,0),
          1,8,
          false);


}

int getmin(int a, int b, int c){

        if(a < b && a < c) return a;
        if(b < a && b < c) return b;
        return c;

}

void trackbarchange(int, void*){

        std::ostringstream oss;
        oss << seg12 << " " << seg13 << " " << seg23;
        writeText(oss,60,60);


        //1. circle detect
        vector<Vec3f> circles;
        HoughCircles( imagegray, circles, CV_HOUGH_GRADIENT, 1, imagegray.rows/8, cannythreshmax, centerthresh, rmin, rmax );
        for( size_t i = 0; i < circles.size(); i++ )
        {
         Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
         int radius = cvRound(circles[i][2]);
         circle( image, center, 3, Scalar(0,255,0), -1, 8, 0 );
         circle( image, center, radius, Scalar(0,0,255), 3, 8, 0 );
         int isredetect = 0;
         for( size_t j = 0; j < circlespm.size(); j++ ){

                 if(dist(center,Point(circlespm[j].x, circlespm[j].y)) < 80){

                         circlespm[j].redetects++;
                         circlespm[j].x = center.x;
                         circlespm[j].y = center.y;
                         isredetect = 1;
                 }

         }
         if(!isredetect){

                 recircle r (circles[i][0],circles[i][1],circles[i][2]);
                 circlespm.push_back( r );

         }

        }

        if(circlespm.empty()) return;

        //prinzipiell könnten kreise durch eg movement updates näher aneinander geraten
        //iterieren und close circles löschen mit gering
        /*for( size_t i = 0; i < circlespm.size(); i++ ){

                for( size_t j = 0; j < circlespm.size(); j++ ){

                        Point p1(cvRound(circlespm[i].x), cvRound(circlespm[i].y));
                        Point p2(cvRound(circlespm[j].x), cvRound(circlespm[j].y));

                        if( dist(p1, p2) < 10){

                                if(  circlespm[i].redetects > circlespm[j].redetects)
                                                circlespm.erase(circlespm.begin()+j);
                                else
                                        circlespm.erase(circlespm.begin()+i);
                        }

                }
        }*/


        //2. bestimmung max
        maxhredetects = 0;
        for( size_t j = 0; j < circlespm.size(); j++ ){

                std::ostringstream oss;
                oss << "x:" << circlespm[j].x <<"y:" << circlespm[j].y <<"det:" << circlespm[j].redetects << " " << circlespm[j].typ << "\n";
                writeText(oss,circlespm[j].x,circlespm[j].y);

                circlespm[j].typ = 0;

                if(circlespm[j].redetects > maxhredetects){
                        maxhidx = j;
                        maxhredetects = circlespm[j].redetects;
                }

                circlespm[j].redetects=circlespm[j].redetects-maxhredetects3+33;

                if(circlespm[j].redetects<-3){

                        circlespm.erase(circlespm.begin()+j);
                }

        }
        circle( image, Point(cvRound(circlespm[maxhidx].x), cvRound(circlespm[maxhidx].y)), cvRound(circlespm[maxhidx].r), Scalar(100,100,255), 6, 8, 0 );
        std::ostringstream oss2;
        oss2 << "max";
        writeText(oss2, circlespm[maxhidx].x-60, circlespm[maxhidx].y);

        int maxhredetects2 = 0;
        size_t maxhidx2 = 0;
        for( size_t j = 0; j < circlespm.size(); j++ ){

                        if(circlespm[j].redetects > maxhredetects2
                                        && dist(Point(circlespm[j].x, circlespm[j].y),Point(circlespm[maxhidx].x, circlespm[maxhidx].y))>2){
                                maxhidx2 = j;
                                maxhredetects2 = circlespm[j].redetects;
                        }

                }
        circle( image, Point(cvRound(circlespm[maxhidx2].x), cvRound(circlespm[maxhidx2].y)), cvRound(circlespm[maxhidx2].r), Scalar(100,100,255), 6, 8, 0 );
        std::ostringstream oss3; oss3 << "max2";
        writeText(oss3, circlespm[maxhidx2].x, circlespm[maxhidx2].y-60);

        maxhredetects3 = 0;
        size_t maxhidx3 = 0;
                for( size_t j = 0; j < circlespm.size(); j++ ){

                                if( (circlespm[j].redetects > maxhredetects3)
                                                && (dist(Point(circlespm[j].x, circlespm[j].y),Point(circlespm[maxhidx].x, circlespm[maxhidx].y))>2)
                                                && (dist(Point(circlespm[j].x, circlespm[j].y),Point(circlespm[maxhidx2].x, circlespm[maxhidx2].y))>2) ){
                                        maxhidx3 = j;
                                        maxhredetects3 = circlespm[j].redetects;
                                }

                        }
        circle( image, Point(cvRound(circlespm[maxhidx3].x), cvRound(circlespm[maxhidx3].y)), cvRound(circlespm[maxhidx3].r), Scalar(100,100,255), 6, 8, 0 );
        std::ostringstream oss4; oss4 << "max3";
        writeText(oss4, circlespm[maxhidx3].x-60, circlespm[maxhidx3].y-60);

        //3. Bounding Box
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        Canny( imagegray, canny_output, cthreshl, cthreshh, 3 );
        findContours( canny_output, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
        vector<vector<Point> > contoursApprox( contours.size() );
        vector<bool > allowedToDraw( contours.size() );
        vector<RotatedRect> boundRect( contours.size() );
        vector<Point2f>center( contours.size() );
        vector<float>radius( contours.size() );
        Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
        for(size_t i = 0; i<contours.size(); i++){//ruckelt langsam approx poly notwendig
                        //approxPolyDP(contours[i],contoursApprox[i], (float)maxapproxdist/100, 1); //always CV_POLY_APPROX_DP
                        boundRect[i] = minAreaRect( Mat(contours[i]) );//boundingRect
                        //minEnclosingCircle( (Mat)contoursApprox[i], center[i], radius[i] );
                }
        for(size_t i=0; i<boundRect.size(); i++){

                RotatedRect curRectangle = boundRect[i];
                if( (abs(curRectangle.size.height - minbbheight)<50 && abs(curRectangle.size.width - minbbwidth)<50 ) ||
                                ( abs(curRectangle.size.width - minbbheight)<50 && abs(curRectangle.size.height - minbbwidth)<50) ){

                        Point2f rect_points[4]; boundRect[i].points( rect_points );
                                                  for( int j = 0; j < 4; j++ )
                                                         line( image, rect_points[j], rect_points[(j+1)%4], Scalar(255,0,0), 1, 8 );
                }
        }


        //for each pair of circles check bounding box distance smaller 20
        for( size_t i = 0; i < circlespm.size(); i++ ){

                for( size_t j = 0; j < circlespm.size(); j++ ){

                        Point p1(cvRound(circlespm[i].x), cvRound(circlespm[i].y));
                        Point p2(cvRound(circlespm[j].x), cvRound(circlespm[j].y));
                        Point segcenter( (p1.x+p2.x)/2, (p1.y+p2.y)/2 );

                        for(size_t k=0; k<boundRect.size(); k++){

                                RotatedRect curRectangle = boundRect[k];
                                if( (abs(curRectangle.size.height - minbbheight)<30 && abs(curRectangle.size.width - minbbwidth)<30 ) ||
                                        ( abs(curRectangle.size.width - minbbheight)<30 && abs(curRectangle.size.height - minbbwidth)<30) ){

                                        if( dist(curRectangle.center, segcenter) < 40 && dist(p1,p2) < 270 && dist(p1,p2) > 210){ //werte mit scalar factor parametrisieren um zoombar zu machen wie bei Antennendetektion

                                                line( image, p1, p2, Scalar(0,200,0), 3, 8 );
                                                circlespm[i].redetects++;
                                                circlespm[j].redetects++;

                                                std::ostringstream oss;
                                                oss << dist(p1,p2);
                                                writeText(oss,(p1.x+p2.x)/2, (p1.y+p2.y)/2);

                                        }

                                }

                        }

                }
        }


        //check distance parameters 2
        for( size_t i = 0; i < circlespm.size(); i++ ){

                        for( size_t j = 0; j < circlespm.size(); j++ ){

                                Point p1(cvRound(circlespm[i].x), cvRound(circlespm[i].y));
                                Point p2(cvRound(circlespm[j].x), cvRound(circlespm[j].y));
                                Point segcenter( (p1.x+p2.x)/2, (p1.y+p2.y)/2 );

                                for(size_t k=0; k<boundRect.size(); k++){

                                        RotatedRect curRectangle = boundRect[k];
                                        if( (abs(curRectangle.size.height - minbbheight2)<30 && abs(curRectangle.size.width - minbbwidth2)<30 ) ||
                                                ( abs(curRectangle.size.width - minbbheight2)<30 && abs(curRectangle.size.height - minbbwidth2)<30) ){

                                                if( dist(curRectangle.center, segcenter) < 40 && dist(p1,p2) < 270 && dist(p1,p2) > 210 ){

                                                        line( image, p1, p2, Scalar(0,200,0), 3, 8 );
                                                        //bonus for both endpoints
                                                        circlespm[i].redetects++;
                                                        circlespm[j].redetects++;

                                                        if(dist(p1, Point(circlespm[maxhidx].x,circlespm[maxhidx].y ) ) < 10 && dist(p2, Point(circlespm[maxhidx2].x,circlespm[maxhidx2].y ) ) < 10){

                                                                if(seg12 < 500) seg12+=1.5;

                                                        }

                                                        if(dist(p2, Point(circlespm[maxhidx].x,circlespm[maxhidx].y ) ) < 10 && dist(p1, Point(circlespm[maxhidx2].x,circlespm[maxhidx2].y ) ) < 10){

                                                                if(seg12 < 500) seg12+=1.5;

                                                        }

                                                        if(dist(p1, Point(circlespm[maxhidx2].x,circlespm[maxhidx2].y ) ) < 10 && dist(p2, Point(circlespm[maxhidx3].x,circlespm[maxhidx3].y ) ) < 10){

                                                                if(seg23 < 500) seg23+=1.5;

                                                        }

                                                        if(dist(p1, Point(circlespm[maxhidx3].x,circlespm[maxhidx3].y ) ) < 10 && dist(p2, Point(circlespm[maxhidx2].x,circlespm[maxhidx2].y ) ) < 10){

                                                                if(seg23 < 500) seg23+=1.5;

                                                        }

                                                        if(dist(p1, Point(circlespm[maxhidx].x,circlespm[maxhidx].y ) ) < 10 && dist(p2, Point(circlespm[maxhidx3].x,circlespm[maxhidx3].y ) ) < 10){

                                                                if(seg13 < 500) seg13+=1.5;

                                                        }

                                                        if(dist(p1, Point(circlespm[maxhidx3].x,circlespm[maxhidx3].y ) ) < 10 && dist(p2, Point(circlespm[maxhidx].x,circlespm[maxhidx].y ) ) < 10){

                                                                if(seg13 < 500) seg13+=1.5;

                                                        }

                                                        std::ostringstream oss;
                                                        oss << dist(p1,p2);
                                                        writeText(oss,(p1.x+p2.x)/2, (p1.y+p2.y)/2);


                                                }

                                        }

                                }

                        }
                }


        if(seg12 > 0) seg12 --; //getmin(seg12, seg13, seg23);
        if(seg23 > 0) seg23--; //getmin(seg12, seg13, seg23);
        if(seg13 > 0) seg13--;///getmin(seg12, seg13, seg23);

        /*line( image, Point(circlespm[maxhidx].x,circlespm[maxhidx].y),
                        Point(circlespm[maxhidx2].x,circlespm[maxhidx2].y),
                        Scalar(30,255,30), 15, 8 );
        line( image, Point(circlespm[maxhidx].x,circlespm[maxhidx].y),
                                Point(circlespm[maxhidx3].x,circlespm[maxhidx3].y),
                                Scalar(30,255,30), 15, 7 );*/

        if( abs(circlespm[maxhidx].redetects-circlespm[maxhidx2].redetects)>500 && abs(circlespm[maxhidx].redetects-circlespm[maxhidx3].redetects)>500 && abs(circlespm[maxhidx2].redetects-circlespm[maxhidx3].redetects)>500   )
        {

                if(seg12 < seg23 && seg12 < seg13){

                                if(dist(Point(circlespm[maxhidx].x, circlespm[maxhidx].y),Point(circlespm[maxhidx2].x, circlespm[maxhidx2].y))>237 )
                                                                line( image, Point(circlespm[maxhidx].x,circlespm[maxhidx].y),
                                                                                        Point(circlespm[maxhidx2].x,circlespm[maxhidx2].y),
                                                                                        Scalar(30,255,30), 1, 8 );
                                else
                                        line( image, Point(circlespm[maxhidx].x,circlespm[maxhidx].y),
                                                                                Point(circlespm[maxhidx2].x,circlespm[maxhidx2].y),
                                                                                Scalar(0,0,255), 15, 8 );

                        }

                        if(seg23 < seg12 && seg23 < seg13){

                                        if(dist(Point(circlespm[maxhidx2].x, circlespm[maxhidx2].y),Point(circlespm[maxhidx3].x, circlespm[maxhidx3].y))>237 )
                                                                        line( image, Point(circlespm[maxhidx2].x,circlespm[maxhidx2].y),
                                                                                                Point(circlespm[maxhidx3].x,circlespm[maxhidx3].y),
                                                                                                Scalar(30,255,30), 1, 8 );
                                        else
                                                line( image, Point(circlespm[maxhidx2].x,circlespm[maxhidx2].y),
                                                                                        Point(circlespm[maxhidx3].x,circlespm[maxhidx3].y),
                                                                                        Scalar(0,0,255), 15, 8 );

                                }

                        if(seg13 < seg12 && seg13 < seg23){

                                                if(dist(Point(circlespm[maxhidx].x, circlespm[maxhidx].y),Point(circlespm[maxhidx3].x, circlespm[maxhidx3].y))>237 )
                                                                                line( image, Point(circlespm[maxhidx].x,circlespm[maxhidx].y),
                                                                                                        Point(circlespm[maxhidx3].x,circlespm[maxhidx3].y),
                                                                                                        Scalar(30,255,30), 1, 8 );
                                                else
                                                        line( image, Point(circlespm[maxhidx].x,circlespm[maxhidx].y),
                                                                                                Point(circlespm[maxhidx3].x,circlespm[maxhidx3].y),
                                                                                                Scalar(0,0,255), 15, 8 );

                                        }

        }





        /*
        if(dist(Point(circlespm[maxhidx].x, circlespm[maxhidx].y),Point(circlespm[maxhidx3].x, circlespm[maxhidx3].y))>220 )
                        line( image, Point(circlespm[maxhidx].x,circlespm[maxhidx].y),
                                                Point(circlespm[maxhidx3].x,circlespm[maxhidx3].y),
                                                Scalar(30,255,30), 1, 8 );
        else
                line( image, Point(circlespm[maxhidx].x,circlespm[maxhidx].y),
                                                        Point(circlespm[maxhidx3].x,circlespm[maxhidx3].y),
                                                        Scalar(0,0,255), 15, 8 );



        if(dist(Point(circlespm[maxhidx2].x, circlespm[maxhidx2].y),Point(circlespm[maxhidx3].x, circlespm[maxhidx3].y))>220 )
                line( image, Point(circlespm[maxhidx2].x,circlespm[maxhidx2].y),
                                        Point(circlespm[maxhidx3].x,circlespm[maxhidx3].y),
                                        Scalar(30,255,30), 1, 8 );
        else
                line( image, Point(circlespm[maxhidx2].x,circlespm[maxhidx2].y),
                                                        Point(circlespm[maxhidx3].x,circlespm[maxhidx3].y),
                                                        Scalar(0,0,255), 15, 8 );*/

        //merge close points of model to avoid tracking two after disturbance
        //move one with higher redetect to other pos
        //keep higher redetect one so in next round its position is updated
        /*for( size_t i = 0; i < circlespm.size(); i++ ){x

                                for( size_t j = 0; j < circlespm.size(); j++ ){

                                        Point p1(cvRound(circlespm[i].x), cvRound(circlespm[i].y));
                                        Point p2(cvRound(circlespm[j].x), cvRound(circlespm[j].y));

                                        if( dist(p1, p2) < 40){

                                                if(  circlespm[i].redetects > circlespm[j].redetects)
                                                                circlespm.erase(circlespm.begin()+j);
                                                else
                                                        circlespm.erase(circlespm.begin()+i);
                                        }

                                }
        }*/


        //may write to each (new collision) line distance


}

int main( int argc, char** argv )
{

        //VideoCapture vc1 = VideoCapture("videos/robotArm_sticker_1_mid_26_03.ogv");
        namedWindow("img",CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
        namedWindow("img2",CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
        namedWindow("trackbars",CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
        createTrackbar("rmin","trackbars", &rmin, 200, trackbarchange, 0);
        createTrackbar("rmax","trackbars", &rmax, 200, trackbarchange, 0);
        createTrackbar("cannythreshmax","trackbars", &cannythreshmax, 200, trackbarchange, 0);
        createTrackbar("centerthresh","trackbars", &centerthresh, 200, trackbarchange, 0);
        createTrackbar("cthreshl","trackbars", &cthreshl, 500, trackbarchange, 0);
        createTrackbar("cthreshh","trackbars", &cthreshh, 500, trackbarchange, 0);
        createTrackbar("maxapproxdist","trackbars", &maxapproxdist, 200, trackbarchange, 0);
        createTrackbar("minbbheight","trackbars", &minbbheight, 500, trackbarchange, 0);
        createTrackbar("minbbwidth","trackbars", &minbbwidth, 500, trackbarchange, 0);
        createTrackbar("minbbheight2","trackbars", &minbbheight2, 500, trackbarchange, 0);
        createTrackbar("minbbwidth2","trackbars", &minbbwidth2, 500, trackbarchange, 0);


        Mat grad;
        int ddepth = CV_16S;
        int scale = 1;
        int delta = 0;
        Mat grad_x, grad_y;
        Mat abs_grad_x, abs_grad_y;

        VideoCapture video = VideoCapture("videos/robotArmTUCreal_CLIPCHAMP_keep_CLIPCHAMP_480p.ogv");
    	if(!video.isOpened()){					// if video file is missing
    		cout << "Video not found" << endl;
    		return 1;
    	}
    	VideoWriter outputVideo = VideoWriter();
    	/*
        VideoCapture vc1 = VideoCapture("videos/robotArm_sticker_1_mid_26_03.ogv");
        VideoWriter outputVideo = VideoWriter();
          //int ex = static_cast<int>(vc1.get(CV_CAP_PROP_FOURCC));
          outputVideo.open("out.avi",
                          CV_FOURCC('P','I','M','1'),
                                          vc1.get(CV_CAP_PROP_FPS),
                                          Size((int)vc1.get(CV_CAP_PROP_FRAME_WIDTH),(int)vc1.get(CV_CAP_PROP_FRAME_HEIGHT)),
                                          1);
		*/
          //while(video.read(image))
        while(video.read(image)){

                Sobel( image, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
                convertScaleAbs( grad_x, abs_grad_x );
                Sobel( image, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
                convertScaleAbs( grad_y, abs_grad_y );
                addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, image );

                cvtColor( image, imagegray, CV_RGB2GRAY );

                trackbarchange(0, 0);

                if(!canny_output.empty()) imshow( "img", canny_output);
                imshow( "img2", image);

                outputVideo.write(image);

                //waitKey(1);


        }


}
/////// daniel last email end

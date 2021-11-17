#include <iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<chrono>
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include<fstream>
#include<sstream>
#include<pangolin/pangolin.h>
#include<vector>
#include<Eigen/Core>
#include<Eigen/Dense>

#include <unistd.h>



using namespace std;
using namespace cv;
using namespace Eigen;



Point2d pixel2cam(const Point &p, const Mat &K);
double testTriangulation(vector<Point> &ptsL,vector<Point> &ptsR,Mat_<double> R, Mat_<double> t);



static string img_pathL = "/home/zhaoqiang/3D_project/play0/";
static string img_pathR = "/home/zhaoqiang/3D_project/play1/";
static string imgrequire_pathL = "/home/zhaoqiang/3D_project/play0_require/";
static string imgrequire_pathR = "/home/zhaoqiang/3D_project/play1_require/";
static string out_3dpoint = "/home/zhaoqiang/3D_project/3DPoint.txt";
static int img_num = 2880;
static int img_require_num = 1300;





static void on_mouseL( int event, int x, int y, int flags, void* ustc)  
{
    
    //     写入文件
    ofstream outfile;
    
    Point2d pt;
    
    
    Mat& src = *(Mat*)ustc;
    
        
    if(event == CV_EVENT_LBUTTONDOWN)//鼠标左键按下事件发生  
    {

        outfile.open("./point1.txt", ios::app);
        pt.x = (double)x;
        pt.y = (double)y;
//         打印当前坐标值
        string res1, res2;
        stringstream ss1, ss2;
        ss1 << x;
        ss2 << y;
        ss1 >> res1;
        ss2 >> res2;
        
        
        string temp = "("+res1+","+res2+")";
 

        int font_face = FONT_HERSHEY_SIMPLEX;

        putText(src,temp, pt, font_face, 1,  Scalar(255, 255, 255), 1,8); //在图像中打印当前坐标值 
        
        outfile << pt.x <<" " <<pt.y  << endl;
        circle( src, pt, 2, Scalar(255,0,0) ,1, 8);//在在图像当前坐标点下画圆  
        imshow( "左视图choose point", src );
        outfile.close();
        
    }    
    
} 

static void on_mouseR( int event, int x, int y, int flags, void* ustc)  
{
    
    //     写入文件
    ofstream outfile;
    Point2d pt;
    
    Mat& src = *(Mat*)ustc;
    if(event == CV_EVENT_LBUTTONDOWN)//鼠标左键按下事件发生  
    {

        outfile.open("./point2.txt", ios::app);
        pt.x = (double)x;
        pt.y = (double)y;
//         打印当前坐标值
        string res1, res2;
        stringstream ss1, ss2;
        ss1 << x;
        ss2 << y;
        ss1 >> res1;
        ss2 >> res2;
        
        string temp = "("+res1+","+res2+")";
        int font_face = FONT_HERSHEY_SIMPLEX;
        putText(src,temp, pt, font_face, 1,  Scalar(255, 255, 255), 1,8); //在图像中打印当前坐标值 
        
        
        outfile << pt.x <<" " <<pt.y  << endl;
        circle( src, pt, 2, Scalar(255,0,0) ,1, 8);//在在图像当前坐标点下画圆  
        imshow( "右视图choose point", src );
        outfile.close();
        
    } 
    
} 


static void on_mouseL_require( int event, int x, int y, int flags, void* ustc)  
{
    
    //     写入文件
    ofstream outfile;
    Point2d pt;
    Mat& src = *(Mat*)ustc;
    if(event == CV_EVENT_LBUTTONDOWN)//鼠标左键按下事件发生  
    {

        outfile.open("./point3.txt", ios::app);
        pt.x = (double)x;
        pt.y = (double)y;
        outfile << pt.x <<" " <<pt.y  << endl;
        circle( src, pt, 2, Scalar(255,0,0) ,1, 8);//在在图像当前坐标点下画圆  
        imshow( "需要恢复的点（左视图）", src );
        outfile.close();
        
    } 
}

static void on_mouseR_require( int event, int x, int y, int flags, void* ustc)  
{
    
    //     写入文件
    ofstream outfile;
    
    Point2d pt;
    
    
    Mat& src = *(Mat*)ustc;
    
    if(event == CV_EVENT_LBUTTONDOWN)//鼠标左键按下事件发生  
    {
        outfile.open("./point4.txt", ios::app);
        pt.x = (double)x;
        pt.y = (double)y;
        outfile << pt.x <<" " <<pt.y  << endl;
        circle( src, pt, 2, Scalar(255,0,0) ,1, 8);//在在图像当前坐标点下画圆  
        imshow( "需要恢复的点（右视图）", src );
        outfile.close();
        
    } 
}


void get_require_points(string image_pathL, string image_pathR)
{
    for(int i = 1; i < img_require_num; i+=100)
    {
    
   
        string is;
        stringstream ss;
        ss << i;
        ss >> is;
        
        string imgfile_pathL = image_pathL+"5-"+is+".jpg";
        string imgfile_pathR = image_pathR+"5-"+is + ".jpg";
        Mat srcL = imread(imgfile_pathL, 1);
        Mat srcR = imread(imgfile_pathR, 1);

        if (srcL.empty() or srcR.empty())
        {
            printf("could not load image " );
            
        }
        namedWindow("需要恢复的点（左视图）", 1);
        namedWindow("需要恢复的点（右视图）", 1);
        setMouseCallback("需要恢复的点（右视图）", on_mouseR_require, (void*)&srcR);
        
        setMouseCallback("需要恢复的点（左视图）", on_mouseL_require, (void*)&srcL);
        
        imshow("需要恢复的点（右视图）", srcR);
        imshow("需要恢复的点（左视图）", srcL);
        int c = waitKey(100);
        if(c == 27)
        {
            break;
        }
        waitKey();
    }
    
        
        
        destroyAllWindows();
//     cvReleaseImage(&src);
}

void get_points(string image_pathL, string image_pathR)
{
    for(int i = 1; i < img_num; i+=20)
    {
        string is;
        stringstream ss;
        ss << i;
        ss >> is;
        
        string imgfile_pathR = image_pathR+"2-"+is+".jpg";
        string imgfile_pathL = image_pathL+"2-"+is+".jpg";
        Mat srcR = imread(imgfile_pathR, 1);
        Mat srcL = imread(imgfile_pathL, 1);
   
   
        if (srcR.empty() or srcL.empty())
        {
            printf("could not load image " );
            
        }
        namedWindow("右视图choose point", 1);
        namedWindow("左视图choose point", 1);
        setMouseCallback("右视图choose point", on_mouseR, (void*)&srcR);
        setMouseCallback("左视图choose point", on_mouseL, (void*)&srcL);
        
        imshow("左视图choose point", srcL);
        imshow("右视图choose point", srcR);
        int c = waitKey(100);
        if(c == 27)
        {
            break;
        }
        waitKey();
    }
    
    destroyAllWindows();
//     cvReleaseImage(&src);
}

    
    
    

void print_points_vector(vector<Point> &pts)
{
    for(vector<Point>::iterator it = pts.begin(); it != pts.end(); it++)
    {
        cout  << (*it) << endl;
    }
}


void pose_estimation_2d2d(vector<Point> &ptsl, vector<Point> &ptsr, Mat &R, Mat &t) {
  // 相机内参,TUM Freiburg2
  Mat K1 = (Mat_<double>(3, 3) << 1460.065, 0, 959.500, 0, 1460.065, 539.500, 0, 0, 1);
  Mat K2 = (Mat_<double>(3, 3) << 1650.115, 0, 959.500, 0, 1650.115, 539.500, 0, 0, 1);
  Mat R1, R2;
  



  

  //-- 计算本质矩阵
  Point2d principal_point(959.500, 539.500);        //相机主点

  Mat essential_matrix, fundamental_matrix;
  fundamental_matrix = findFundamentalMat(ptsl, ptsr, FM_RANSAC);
  essential_matrix = K1.t()*fundamental_matrix*K2;
  
  
  
  cout << "本质矩阵为： " <<endl;
  cout << essential_matrix << endl;
  

  //-- 从本质矩阵中恢复旋转和平移信息.
//   recoverPose(essential_matrix, ptsl, ptsr, R, t, focal_length, principal_point);
  
  decomposeEssentialMat(essential_matrix, R1, R2, t);
  
  cout << "分解结果： " << endl;
  cout << R1 << endl;
  cout << " ***************" << endl;
  
  cout << R2 <<endl;
  cout << " ***************" << endl;
  cout << t << endl;
  
  if(determinant(R1)+1.0 < 1e-09)
  {
       essential_matrix = -essential_matrix;
       decomposeEssentialMat(essential_matrix, R1, R2, t);
  }
//   cout << "分解结果： " << endl;
//   cout << R1 << endl;
//   cout << " ***************" << endl;
//   
//   cout << R2 <<endl;
//   cout << " ***************" << endl;
//   cout << t << endl;
//   筛选都具有正深度的解

        //通过三角化得到的正深度比例选择Rt解
  
//   cout<<"test输出" <<endl;
//   cout << testTriangulation(ptsl, ptsr, R1, t) << endl;
  
  double ratio1 = max(testTriangulation(ptsl, ptsr, R1, t), testTriangulation(ptsl, ptsr, R1, -t));
  double ratio2 = max(testTriangulation(ptsl, ptsr, R2, t), testTriangulation(ptsl, ptsr, R2, -t));
  R= ratio1 > ratio2 ? R1 : R2;

  cout << "旋转矩阵：" << endl;
  cout << R << endl;
    
      
  
  
}



vector<Point> read_points(string &filename, vector<Point> &pts)
{
    ifstream file1;
    file1.open(filename, ios::in);
    
    if(!file1.is_open())
    {
        cout << "文件打开失败" << endl;
    }

    
    double a1L, a2L;
    int L = 1;
//     vector<Point2d>ptsL;
    while (file1 >> a1L >> a2L){	
    //对相应的提取出的每行a1L，a2L进行处理，若需要的话可以将处理之后的数据进行压入到指定文件	
        pts.push_back(Point2d(a1L, a2L));
        L++;
    }
//     for(vector<Point2d>::iterator it = pts.begin(); it != pts.end(); it++)
//     {
//         cout << (*it) << endl;
//     }

    return pts;
    

}
// 计算两个三维点之间的距离
double getDistance(const Point3f &pt1, const Point3f &pt2)
{
    double distance = sqrtf(powf((pt1.x-pt2.x), 2)+powf((pt1.y-pt2.y), 2)+powf((pt1.z-pt2.z), 2));
    return distance;
}
// 计算两个二维点之间的距离
double getdistance2D(const Point2d &pt1, const Point2d &pt2)
{
    double distance = sqrtf(powf((pt1.x-pt2.x), 2)+powf((pt1.y-pt2.y), 2));
    return distance;
    
}



// 根据得到的基础矩阵进行三角剖分

void triangulation(vector<Point>ptsL, vector<Point>ptsR, Mat &R, const Mat &t, vector<Point3d> &points, vector<double> &distance) {
    
//     定义的两个视角的投影矩阵
    
  Mat T1 = (Mat_<double>(3, 4) <<
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0);
  Mat T2 = (Mat_<double>(3, 4) <<
    R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
    R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
    R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)
  );

  Mat K1 = (Mat_<double>(3, 3) << 1460.065, 0, 959.500, 0, 1460.065, 539.500, 0, 0, 1);
  Mat K2 = (Mat_<double>(3, 3) << 1650.115, 0, 959.500, 0, 1650.115, 539.500, 0, 0, 1);
//   vector<Point2f> pts_1, pts_2;
  vector<Point2d> points_camL, points_camR;

  for(vector<Point>::iterator it = ptsL.begin(); it!=ptsL.end(); it++)
  {
      Point2d p_camL = pixel2cam((*it), K1);
      points_camL.push_back(p_camL);
      
  }
  cout << "转换后的左机坐标" <<endl;
  
  for(vector<Point2d>::iterator it = points_camL.begin(); it != points_camL.end(); it++)
  {
        cout  << (*it) << endl;
  }
  for(vector<Point>::iterator it = ptsR.begin(); it!=ptsR.end(); it++)
  {
      Point2d p_camR = pixel2cam((*it), K2);
      
      points_camR.push_back(p_camR);
      
  }
  
  
      
  Mat pts_4d;
  cv::triangulatePoints(T1, T2, points_camL, points_camR, pts_4d);   //相机坐标系下直接到三维
  
  

  // 转换成非齐次坐标
  for (int i = 0; i < pts_4d.cols; i++) {
    Mat x = pts_4d.col(i);
    x /= x.at<double>(3, 0); // 归一化 x = x/()
    Point3d p(
      x.at<double>(0, 0),
      x.at<double>(1, 0),
      x.at<double>(2, 0)
    );
    points.push_back(p);
    
    
  }
  
  for (int i = 0; i < points.size(); i+=2)
  {
      distance.push_back(getDistance(points[i], points[i+1]));
  
  }
  for (vector<double>::iterator it = distance.begin(); it != distance.end(); it++)
  {
      cout << "两点(3d)之间的距离为:  " << (*it) << endl;
  }
  
  
//   cout << "点坐标 " << points << endl;
  
  
}

double testTriangulation(vector<Point> &ptsL, vector<Point> &ptsR, Mat_<double> R, Mat_<double> t)
{
    cv::Mat pointcloud;
    Mat P = (Mat_<double>(3, 4) <<
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0);
    Mat P1 = (Mat_<double>(3, 4) <<
    R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
    R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
    R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)
  );
    vector<Point2d> points_camL, points_camR;
    Mat K1 = (Mat_<double>(3, 3) << 1490.737, 0, 959.500, 0, 1490.737, 539.500, 0, 0, 1);
    Mat K2 = (Mat_<double>(3, 3) << 2471.357, 0, 1004.742, 0, 2471.357, 530.420, 0, 0, 1);
    
    
//     先转换为相机坐标系下
    
    for(vector<Point>::iterator it = ptsL.begin(); it!=ptsL.end(); it++)
    {
        Point2d p_camL = pixel2cam((*it), K1);
        points_camL.push_back(p_camL);
        
    }
    for(vector<Point>::iterator it = ptsR.begin(); it!=ptsR.end(); it++)
    {
        Point2d p_camR = pixel2cam((*it), K2);
        
        points_camR.push_back(p_camR);
        
    }

    triangulatePoints(P, P1, points_camL, points_camR, pointcloud);// 三角化得到路标3D坐标
    
    
//     cout << "test3d点坐标 " << endl;
//     cout << pointcloud <<endl;
//     cout << pointcloud.cols <<endl;
//     cout << pointcloud.rows <<endl;
//     cout << "*****************" <<endl;
//     
//     double normal_factor = pointcloud.col(i).at<double>(3);
//     cout << normal_factor << endl;
//     
//     cout << P.type() << endl;
//     cout<< pointcloud.col(i).type() << endl;
    
/*    
    cout << "*****************" <<endl;
    cout << P * (pointcloud.col(i) / normal_factor) << endl;
    cout << "*****************" <<endl;
    cout << P1 * (pointcloud.col(i) / normal_factor)<< endl;
    */
    
    int front_count = 0;
    for (int i = 0; i < pointcloud.cols; i++)
    {
        double normal_factor = pointcloud.col(i).at<double>(3);
 
        cv::Mat_<double> p_3d_l = cv::Mat(P) * (pointcloud.col(i) / normal_factor);
        cv::Mat_<double> p_3d_r = cv::Mat(P1) * (pointcloud.col(i) / normal_factor);
        if (p_3d_l(2) > 0 && p_3d_r(2) > 0)
            front_count++;
    }
    return 1.0 * front_count / pointcloud.cols;
}




// 将像素坐标系转换为相机坐标系(归一化后)

Point2d pixel2cam(const Point &p, const Mat &K) {
  return Point2d
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

// 验证在归一化相机坐标系下的像素坐标与投影坐标的差别
void test_projection(const vector<Point> &pts_requireL, const vector<Point> &pts_requireR, const vector<Point3d> &points, Mat &R, Mat &t)
{
    
    Mat K1 = (Mat_<double>(3, 3) << 1490.737, 0, 959.500, 0, 1490.737, 539.500, 0, 0, 1);
    Mat K2 = (Mat_<double>(3, 3) << 2471.357, 0, 1004.742, 0, 2471.357, 530.420, 0, 0, 1);
    
    for (int i = 0; i < points.size(); i++)
    {
        Point2d ptsL_cam = pixel2cam(pts_requireL.at(i), K1);
        cout << "左机归一化相机坐标： " << ptsL_cam << endl;
        
        Point2d ptsL_cam_proj(points[i].x/points[i].z, points[i].y/points[i].z);
        cout << "左机投影坐标： " << ptsL_cam_proj << endl;
        
        double distanceL = getdistance2D(ptsL_cam, ptsL_cam_proj);
        cout<< "左机两点(2d)之间的距离为： " << distanceL << endl;
        
        
        
        Point2d ptsR_cam = pixel2cam(pts_requireR.at(i), K2);
        cout << "右机归一化相机坐标： " << ptsR_cam << endl;
        Mat ptsR_trans = R *(Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;
        ptsR_trans /= ptsR_trans.at<double>(2, 0);
        Point2d ptsR_cam_proj(ptsR_trans.at<double>(0, 0), ptsR_trans.at<double>(1, 0));        
        cout << "右机投影坐标： " << ptsR_cam_proj << endl;
        double distanceR = getdistance2D(ptsR_cam, ptsR_cam_proj);
        cout<< "右机两点(2d)之间的距离为： " << distanceR << endl;
    }
}

        
        
        
        


void showPointCloud(const vector<Point3d> &pointcloud) {

    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 750, 480);      //创建大小为1024*768的窗口
    
    glEnable(GL_DEPTH_TEST);    //启动深度测试
    glEnable(GL_BLEND);        //颜色混合
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(752, 480, 420, 420, 320, 320, 0.1, 1000),    // 分辨率+内参+最近最远视距（自己设定）
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, pangolin::AxisY)          //前三个参数为相机的位置，4-6个为相机所看到视点的位置一般为原点， 最后是相机轴的方向
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)   //视图在视窗中的范围  上下左右， 最后一个参数为长宽比
        
        .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
//         glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        
        glLineWidth(3);
        glBegin ( GL_LINES );
	    glColor3f ( 0.8f,0.f,0.f );           //glColor3f(R, G, B)
	    
	    glVertex3f(-0.5, -0.5, -0.5);
	    glVertex3f( 0, -0.5, -0.5 );   //红色  x轴
        
	    glColor3f( 0.f,0.8f,0.f);
	    glVertex3f(-0.5, -0.5, -0.5);    // 绿色
	    glVertex3f( -0.5, 0, -0.5 );
	    glColor3f( 0.2f,0.2f,1.f);
	    glVertex3f(-0.5, -0.5, -0.5);
	    glVertex3f( -0.5, -0.5, 0 );
        glEnd();

        glPointSize(8);
        glBegin(GL_POINTS);
        for (auto &p: pointcloud) {
            glColor3f(1.0, 0.0, 0.0);
            glVertex3d(p.x, p.y, p.z);
        }
        
    
        glEnd();
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
    return;
}

// 将生成的3D坐标写入文件
void write_3Dpoints(string out3Dpath, vector<Point3d>points)
{
    ofstream outfile;
    outfile.open(out3Dpath, ios::out);
    
    for (vector<Point3d>::iterator it = points.begin(); it != points.end(); it++)
    {
        outfile << (*it).x <<" " << (*it).y << " " <<(*it).z << endl;
    }
    outfile.close();
}

    

int main(int argc, char **argv) {
//     std::cout << "Hello, world!" << std::endl;
//     初始化
    
    vector<Point>ptsL, ptsR, pts_reL, pts_reR;
    Mat R, t;
    
//     新建左右机坐标点文件
    FILE *fpL, *fpR, *fp_reL, *fp_reR;
//     fpL = fopen(argv[1], "w");
//     fpR = fopen(argv[2], "w");
    fp_reL = fopen(argv[3], "w");
    fp_reR = fopen(argv[4], "w");
    
    
    
//     先进行类型转换坐标点文件
    string filename1 = argv[1];
    string filename2 = argv[2];
    string filename3 = argv[3];
    string filename4 = argv[4];

//     get_points(img_pathL, img_pathR);

    

    
    
//     读取文件中的点
    read_points(filename1, ptsL);
//     cout << "左机2d坐标: " << endl;
//     print_points_vector(ptsL);
    
    read_points(filename2, ptsR);
//     cout << "右机2d坐标: " << endl;
//     print_points_vector(ptsR);
//     
//     
    pose_estimation_2d2d(ptsL, ptsR, R, t);
    cout << "R=" <<endl;
    cout <<R <<endl;
    cout <<"t = " <<endl;
    cout << t << endl;
    
    
    get_require_points(imgrequire_pathL, imgrequire_pathR);
    read_points(filename3, pts_reL);
    read_points(filename4, pts_reR);
    
    vector<Point3d> points_3D;
    vector<double> distance;
    triangulation(pts_reL, pts_reR, R, t, points_3D, distance);
    
    test_projection(pts_reL, pts_reR, points_3D, R, t);
    
    
    write_3Dpoints(out_3dpoint, points_3D);
    
    
//     生成点云
//     vector<Vector3d, Eigen::aligned_allocator<Vector4d>>pointcloud;
    showPointCloud(points_3D);
    
//     
    
//     
    
    return 0;
}

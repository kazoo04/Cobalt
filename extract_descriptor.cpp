// extract_descriptor.cpp
//
// Usagee Inc. Kazuya Gokita
//

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>

using namespace std;
using namespace cv;

// Maximum image height/width
const int image_size = 100;

// Dense sampling parameters.
//
// See Also:
//  Li Fei-Fei, Pietro Perona, A Bayesian Hierarchical Model for Learning Natural Scene Categories, CVPR 2005.
//  Eric Nowak, Frederic Jurie, and Bill Triggs, Sampling Strategies for Bag-of-Features Image Classification, ECCV 20.
//
const float initFeatureScale = 8.0f;
const int   featureScaleLevels = 3;
const float featureScaleMul = 1.4142f;
const int   initXyStep = 3;
const int   initImgBound = initFeatureScale;

static void help(char *argv[])
{
    cout << "\nUsage: " << argv[0] << " path/to/image\n" << endl;
}

bool load_image(char *filename, Mat &dist)
{
    Mat src = imread(filename);

    if(src.data) {
        double scale = (double)image_size / (src.rows > src.cols ? src.rows : src.cols);
        dist = Mat(src.rows * scale, src.cols * scale, src.type());
        resize(src, dist, Size(), scale, scale);

        return true;
    }
    
    return false;
}

int main(int argc, char *argv[])
{
    if(argc != 2) {
        help(argv);
        return -1;
    }

    Mat img;
    if(!load_image(argv[1], img)) {
        cout << "Error loading image: " << argv[1] << endl;
        return -1;
    }

    vector<KeyPoint> keypoints;

    // Dense detector
    // The detector generates several levels of features.
    // Feature scale, step size, and size of boundary are multiplied by "featureScaleMul".
    DenseFeatureDetector detector(
        initFeatureScale,
        featureScaleLevels,
        featureScaleMul,
        initXyStep,
        initImgBound,
        false,        // varyXyStepWithScale 
        true          // varyImgBoundWithScale
    );

    // Dense sampling
    detector.detect(img, keypoints);

    //descriptor (64d vectors)
    Mat descriptors;

    //extractor
    FREAK extractor;
    extractor.compute(img, keypoints, descriptors);

    //cout << format(descriptors, "csv") << endl;
    
    int num_rows = descriptors.rows;
    int num_columns = descriptors.cols;
    for(int row = 0; row < num_rows; row++) {
        for(int col = 0; col < num_columns; col++) {
            cout << (char)descriptors.data[row * num_columns + col];
        }
    }

    return 0;
}

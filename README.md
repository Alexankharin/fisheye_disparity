# fisheye_disparity
create disparity maps for fisheye and equirect videos
inverse depth mapping for panoramic stero images Used model from https://github.com/gengshan-y/high-res-stereo
Inputs for fisheye_depthmapper.py should be Fisheye or equirect projections and requires properly installed ffmpeg-python wrapper as well as ffmpeg tool. 
Output dispmap. If not specified input FOV is 180

#Installation 
clone or download this repository and install ffmpeg library and add it to PATH variable to be accessable from terminal

install python (reccomended to use Conda virtual enviroment) 

go to folder and install dependencies from requirements.txt

```
conda create -n dispmapper
conda activate dispmapper
pip install -r requirements.txt
```
To run the converter in basic mode run 
```
python fisheye_depthmapper.py --i inputvideoname.mp4 --o outputvideoname.mp4 --resolution 512 --fovx 200 --fovy 200
```

For more info about options enter
```
python fisheye_depthmapper.py --help
```

Resolution should be divisible by 256 for proper work

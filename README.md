# An Effective Semi-Automatic Road Centerline Extraction Method from VHR
### Abastract 
Road digitizing is a labor-intensive task. In order to reduce the cost of human intervention, this paper presents an effective semi-automatic road delineation method based on geodesic distance field and piecewise polygon fitting. The main components of the approach are circular template optimization, soft road center kernel density estimation, fast geodesic distance field generation and piecewise polygon fitting model. First, the optimal circular template is proposed to automatically measure the road width, adjust the manual seeds to road, and further produce the road saliency map according to the local color features inside the starting circle template. the soft road center kernel density directly estimated on road saliency map is introduced to provide flex road center probability which overcomes the difficulty of threshold presetting of road segmentation in traditional road center kernel density estimation. Most importantly, a fast geodesic distance field generation is designed to quickly extract the geodesic between two adjacent seeds, which dramatically increase the efficiency of road tracking comparing to fast marching method. Finally, piecewise polygon fitting model promotes the robustness of our method. Extensive experiments and quantitative comparisons show that the proposed algorithm greatly reduces manual intervention, and significantly improves the overall efficiency of road extraction. Furthermore, the proposed approach takes almost the same time to extract any length of road segment given fixed image size, and no hyperparameters need to be set, which provides good experience in human-computer interaction.

### Dataset
We use [Google Earth](http://www.escience.cn/people/guangliangcheng/Datasets.html) to evaluation the performance of our method.

### Experiment results

#### Extract demo
The videos show the procedure of some test cases.  
<a href='videos/image35.mp4?raw=true'/>Video 1. A tracking process on a image35 (1187KB).(click here)</a>
<br />
<br />
<a href='videos/image36.mp4?raw=true'/>Video 2. A tracking process on a image36 (558KB).(click here)</a>
<br />
<br />
<a href='videos/image102.mp4?raw=true'/>Video 1. A tracking process on a image102 (1157KB).(click here)</a>
<br />
<br />
<a href='videos/image130.mp4?raw=true'/>Video 2. A tracking process on a image130 (1112KB).(click here)</a>

The demonstrations also show that we do not need to set any hyperparameter to run our algorithm.
#### Extraction results
We post some test cases here to demonstrate the effective of our method. The yellow circles in the results are the automatic estimated circular templates. Each of them is estimated according to manual seed.
<p>
    <img src='images/image1.jpg?raw=true' />
</p>
<p>
    <img src='images/image8.png?raw=true' />
</p>
<p>
    <img src='images/image130.jpg?raw=true' />
</p>
<p>
    <img src='images/image13.jpg?raw=true' />
</p>
<p>
    <img src='images/image20.jpg?raw=true' />
</p>
<p>
    <img src='images/image34.jpg?raw=true' />
</p>
<p>
    <img src='images/image92.png?raw=true' />
</p>
<p>
    <img src='images/image141.png?raw=true' />
</p>
<p>
    <img src='images/image155.jpg?raw=true' />
</p>
<p>
    <img src='images/image173.jpg?raw=true' />
</p>

<h3> Times Consuming  </h3>

The experimental operating environment is a 4-core 1.8GHz i7 processor, a notebook with 16G memory, the implementation language is Python3.7, and the iteration parts of the three algorithms are all accelerated by numba  
We Extract three road centerlines of different lengths. As shown in below, the three road sections are AB, AC, and AD, and the road lengths are about 480 meters, 990 meters, and 1310 meters, respectively.
<p>
    <img src='images/image88-2.png?raw=true' />
</p>  
To make data more credible, we extract three times for each segment. The Time Consuming are shown as below  

|segment|Avg.|
|:--- | :---: |
|AB	|0.2568| 
|AC	|0.2753|
|AD	|0.2903|

The data shows that the proposed algorithm takes almost the same time to extract any length of road centerline given a fixed image size.

If our work has any inspiration for your research, please cite our paper:

<pre>
    The paper is undergoing review

</pre>
# Human Pose Estimation from RGB Camera - The repo
In recent years, tremendous amount of progress is being made in the field of Human Pose Estimation from RGB Camera, which is an interdisciplinary field that fuses computer vision, deep/machine learning and anatomy. This repo is for my study notes and will be used as a place for triaging new research papers. 

## Get Involved
To make it more collaberative, hit me up if you see a paper I missed, or if you have a suggestion for something interesting.

## How this is sorted
The papers gotta be sorted by season. This is to make it sane, there are way too many papers out there and this gives some good arbitrary division. Then they're sorted by month then by first letter. Again, arbitrary division to make locating and organizing them easier. This chronologically based layout also helps with because newer papers usually build off of old ones. Ain't nobody care about yo 5 'refinements' to a paper a year after its release. That's why it's better to put the release date of the FIRST instance of the paper. They also have very short and memorable memonic descriptions. Equivariant Siamese Network is no good. Twins look at a guy and compare notes is good. They should be memorable, vivid, and descriptive to the essence of what makes the project different from the others.

- Time Dimension
	- :camera: Single-Shot 
	- :movie_camera: Video/Real-Time
- Spatial Dimensions
	- :door: 2D Models
	- :package: 3D Models

---

## Projects and papers
### Fall 2018

**D3DPEUPBP**

:camera::package:[Deep 3D Human Pose Estimation Under Partial Body Presence](https://ieeexplore.ieee.org/document/8451031) (Oct 2018)

`My legs have been chopped off, and my head, but tis but a scratch.`

**3DHPEUSOIRT**

:camera::package:[3D Human Pose Estimation Using Stochastic Optimization In Real Time](https://www.researchgate.net/profile/Philipp_Werner/publication/327995319_3D_Human_Pose_Estimation_Using_Stochastic_Optimization_in_Real_Time/links/5bc8233992851cae21ad83ac/3D-Human-Pose-Estimation-Using-Stochastic-Optimization-in-Real-Time.pdf) (Oct 2018)

`Try again and again, till you get it right. Uses depth based sensors. RTW + Particle Swarms`

**A3DHPEVMDS**

:camera::package:[Adversarial 3D Human Pose Estimation via Multimodal Depth Supervision](https://arxiv.org/pdf/1809.07921v1.pdf) (Sep 2018)

`Continuation of FBI work, also got multimodal network now. IDK What that means`

**DLCMFHPE**

:camera::door:[Deeply Learned Compositional Models for Human Pose Estimation](http://openaccess.thecvf.com/content_ECCV_2018/papers/Wei_Tang_Deeply_Learned_Compositional_ECCV_2018_paper.pdf) (Sep 2018)

`Take the human body, and shove it into code blocks`

**DPT**

:door:[Dense Pose Transfer](https://arxiv.org/pdf/1809.01995.pdf) (Sep 2018)

`color in a mannequin, a machine imagines the details, then animates it's paper statue`

**3DEPEVIL**

:camera::package:[3D Ego-Pose Estimation via Imitation Learning](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ye_Yuan_3D_Ego-Pose_Estimation_ECCV_2018_paper.pdf) (Sep 2018)

`Headcam, they use a very complicated ragdoll, also, just walking and running`

**3DHPEWSEE**

:camera::package:[3D Human Pose Estimation with Siamese Equivariant Embedding](https://arxiv.org/pdf/1809.07217.pdf) (Sep 2018)

`Twins compare their answers after doing their math homework. (Homework refers to the 3d pose estimation)`

**SOAWVFT2018ECCVPTCO3DHPE**

:camera::package:[Synthetic Occlusion Augmentation with Volumetric Heatmaps for the 2018 ECCV PoseTrack Challenge on 3D Human Pose Estimation](https://arxiv.org/pdf/1809.04987v1.pdf) (Sep 2018)

`They block their face and body with cheap photoshop techniques, then the machine has to "x-ray" through all that.`

### Summer 2018

**NBFUDLAMBHPASE**

:camera::package:[Neural Body Fitting: Unifying Deep Learning and Model-Based Human Pose and Shape Estimation](https://arxiv.org/pdf/1808.05942.pdf) [[CODE]](http://github.com/mohomran/neural_body_fitting) (Aug 2018)

`The circle of 3D pose estimation. 2d Image -> 2d Color Me Rad guy -> 3d pudgy man -> photograph of said man -> 2d Image`

**SSMP3DBPEFMRGBI**

:camera::package: [Single-Shot Multi-Person 3D  Body Pose Estimation From Monocular RGB Input](https://arxiv.org/pdf/1712.03453.pdf) (Aug 2018)

`They use a ORPM, whatever that means. And they have some very obviouly green screened images.`

**RPI3DMSRARFMMC**

:camera::package: [Rethinking Pose in 3D: Multi-stage Refinement and Recovery for Markerless Motion Capture](https://arxiv.org/pdf/1808.01525v1.pdf) (Aug 2018)

`use lots of cameras to make just one camera better, and do this over and over and over again`

**3DHPEWRN**

:camera::package:[3D Human Pose Estimation with Relational Networks](https://arxiv.org/pdf/1805.08961v2.pdf) (Jul 2018)

`Back bone connected to the shoulder bone, shoulder bone connected to the neck bone...`

**HPEWPIL**

:door:[Human Pose Estimation with Parsing Induced Learner](http://openaccess.thecvf.com/content_cvpr_2018/papers/Nie_Human_Pose_Estimation_CVPR_2018_paper.pdf) (Jun 2018)

` `

**FBIPTBTGB2DIA3DHPUFOBI**

:camera::package:[FBI-Pose: Towards Bridging the Gap between 2D Images and 3D Human Poses using Forward-or-Backward Information](https://arxiv.org/pdf/1806.09241) (Jun 2018)

`Anderson Silva's broken bent leg`

### Spring 2018

**DRP3DDRI3DHPE**

:camera::package:[DRPose3D: Depth Ranking in 3D Human Pose Estimation](https://arxiv.org/pdf/1805.08973.pdf) (May 2018)

`These guys do FBI but without the "crowd sourced" annotations`

**IARM3DHPEFWSD**

:camera::package:[It's all Relative: Monocular 3D Human Pose Estimation from Weakly Supervised Data](https://arxiv.org/pdf/1805.06880v2.pdf) (May 2018)

`Crowd sourced relative depth annotations`

:package:[BodyNet: Volumetric Inference of 3D Human Body Shapes](https://arxiv.org/pdf/1804.04875v3.pdf) [[CODE]](https://github.com/gulvarol/bodynet) (Apr 2018)

`Make a statue of michealangelo in minecraft`

:package:[Unsupervised Geometry-Aware Representation for 3D Human Pose Estimation](http://openaccess.thecvf.com/content_ECCV_2018/papers/Helge_Rhodin_Unsupervised_Geometry-Aware_Representation_ECCV_2018_paper.pdf)  [[CODE]](https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning) (Apr 2018)

`It rotates the person with it's eyes. (Insert reference to undressing with eyes here)`

:movie_camera::package: [MonoPerfCap: Human Performance Capture from Monocular Video](http://gvv.mpi-inf.mpg.de/projects/wxu/MonoPerfCap/content/monoperfcap.pdf) [[Project]](http://gvv.mpi-inf.mpg.de/projects/wxu/MonoPerfCap/) (Mar 2018)

`makes a 3d replica of you like a fully featured action figure`

:package:[Learning to Estimate 3D Human Pose and Shape from a Single Color Image](http://openaccess.thecvf.com/content_cvpr_2018/papers/Pavlakos_Learning_to_Estimate_CVPR_2018_paper.pdf) (May 2018)

`SMPL brand Artist's Mannequin`

:camera::package: [3D Human Pose Estimation in the Wild by Adversarial Learning](https://arxiv.org/pdf/1803.09722.pdf) (Mar 2018)

` `

:movie_camera::package: [LCR-Net++: Multi-person 2D and 3D Pose Detection in Natural Images](https://arxiv.org/pdf/1803.00455.pdf) [[Project]](https://thoth.inrialpes.fr/src/LCR-Net/) (Mar 2018)

` `

:camera::package: [Unsupervised Adversarial Learning of 3D Human Pose from 2D Joint Locations](https://arxiv.org/pdf/1803.08244.pdf) [[Project page]](https://nico-opendata.jp/en/casestudy/3dpose_gan/index.html) (Mar 2018)

` `

<a name="Winter 2017"/>

### Winter 2017
:movie_camera::package: [End-to-end Recovery of Human Shape and Pose](https://arxiv.org/pdf/1712.06584.pdf) [[CODE]](https://github.com/akanazawa/hmr) (Dec 2017)

` `

:camera::package:[Exploiting temporal information for 3D human pose estimation](http://openaccess.thecvf.com/content_ECCV_2018/papers/Mir_Rayat_Imtiaz_Hossain_Exploiting_temporal_information_ECCV_2018_paper.pdf) (Nov 2017)

` `

:camera::package: [DensePose: Dense Human Pose Estimation In The Wild](https://arxiv.org/pdf/1802.00434.pdf) [[CODE]](https://github.com/facebookresearch/Densepose) [[Project page]](http://densepose.org) (Feb 2018)

` `

### Fall 2017

:movie_camera::door: [Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://arxiv.org/pdf/1611.08050.pdf) [[CODE]](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) (Apr 2017)

` `

:camera::door: [Adversarial PoseNet: A Structure-aware Convolutional Network for Human Pose Estimation](https://arxiv.org/pdf/1705.00389.pdf) (May 2017)

` `

:camera::package: [A simple yet effective baseline for 3d human pose estimation](https://arxiv.org/pdf/1705.03098.pdf) (Aug 2017) [[CODE]](https://github.com/una-dinosauria/3d-pose-baseline)

` `

:movie_camera::package: [VNect: Real-time 3D Human Pose Estimation with a Single RGB Camera](http://gvv.mpi-inf.mpg.de/projects/VNect/content/VNect_SIGGRAPH2017.pdf) [[CODE]](https://github.com/timctho/VNect-tensorflow) [[Project]](http://gvv.mpi-inf.mpg.de/projects/VNect/) (Jul 2017)

` `

:camera::package: [Lifting from the Deep: Convolutional 3D Pose Estimation from a Single Image](https://arxiv.org/pdf/1701.00295.pdf) (Oct 2017)

` `

:camera::package:[Knowledge-Guided Deep Fractal Neural Networks for Human Pose Estimation](https://arxiv.org/pdf/1705.02407.pdf) [[CODE]](http://github.com/Guanghan/GNet-pose) (Aug 2017)

` `

### 2016
:camera::package: [Learning to Fuse 2D and 3D Image Cues for Monocular Body Pose Estimation](https://arxiv.org/pdf/1611.05708.pdf) (Nov 2016)

` `

:camera::package: [Monocular 3D Human Pose Estimation In The Wild Using Improved CNN Supervision](https://arxiv.org/pdf/1611.09813.pdf) [[Project]](http://gvv.mpi-inf.mpg.de/3dhp-dataset/) (Nov 2016)

` `

:camera::package: [MoCap-guided Data Augmentation for 3D Pose Estimation in the Wild](https://arxiv.org/pdf/1607.02046.pdf) (Oct 2016)

` `

:camera::package:[3D Human Pose Estimation Using Convolutional Neural Networks with 2D Pose Information](https://arxiv.org/pdf/1608.03075.pdf) (Sep 2016)

` `

:camera::package: [Keep it SMPL: Automatic Estimation of 3D Human Pose and Shape from a Single Image](https://arxiv.org/pdf/1607.08128.pdf) (Jul 2016)

` `

:camera::door: [Stacked Hourglass Networks for Human Pose Estimation](https://arxiv.org/pdf/1603.06937.pdf) [[CODE]](https://github.com/umich-vl/pose-hg-demo) (Mar 2016)

` `

:camera::door: [Convolutional Pose Machines](https://arxiv.org/pdf/1602.00134.pdf) [[CODE]](https://github.com/shihenw/convolutional-pose-machines-release) (Jan 2016)

` `

### 2014 & 2015

:movie_camera::package: [Spatio-temporal Matching for Human Pose Estimation](http://www.f-zhou.com/hpe/2014_ECCV_STM.pdf) [[Project]](http://www.f-zhou.com/hpe.html) (Dec 2015)

` `

:movie_camera::package: [Sparseness Meets Deepness: 3D Human Pose Estimation from Monocular Video](https://arxiv.org/pdf/1511.09439.pdf) [[Project]](http://cis.upenn.edu/~xiaowz/monocap.html) (Nov 2015)

` `

---

## DataSets

MS COCO

MPII POSE

Human 3.6M

Human Eva

MPI INF 3DHP

Unite The People

Pose Guided Person Image Generation

A Generative Model of People in Clothing

Deformable GANs for Pose Based Human Image Generatoin

Dense Pose Transfer

[Human3.6M](http://vision.imar.ro/human3.6m/description.php)

[HumanEva](http://humaneva.is.tue.mpg.de/)

[MPI-INF-3DHP](http://gvv.mpi-inf.mpg.de/3dhp-dataset/)

[Unite The People](http://files.is.tuebingen.mpg.de/classner/up/)

[Pose Guided Person Image Generation](https://arxiv.org/pdf/1705.09368.pdf) - [[CODE]](https://github.com/charliememory/Pose-Guided-Person-Image-Generation) - Ma, L., Jia, X., Sun, Q., Schiele, B., Tuytelaars, T., & Gool, L.V. (NIPS 2017)

[A Generative Model of People in Clothing](https://arxiv.org/pdf/1705.04098.pdf) - Lassner, C., Pons-Moll, G., & Gehler, P.V. (ICCV 2017)	

[Deformable GANs for Pose-based Human Image Generation](http://openaccess.thecvf.com/content_cvpr_2018/papers/Siarohin_Deformable_GANs_for_CVPR_2018_paper.pdf) - [[CODE]](https://github.com/AliaksandrSiarohin/pose-gan) - Siarohin, A., Sangineto, E., Lathuilière, S., & Sebe, N. (CVPR 2018)

[Dense Pose Transfer](https://arxiv.org/pdf/1809.01995.pdf) - Neverova, N., Guler, R.A., & Kokkinos, I. (ECCV 2018)

---

## Guide

[Gesture and Sign Language Recognition with Deep Learning](https://biblio.ugent.be/publication/8573066/file/8573068)

[Human Pose Estimation 101](https://github.com/cbsudux/Human-Pose-Estimation-101)

[Bob](https://github.com/Bob130/Human-Pose-Estimation-Papers)

[Jessie](https://github.com/dongzhuoyao/jessiechouuu-adversarial-pose)

[Awesome](https://github.com/cbsudux/awesome-human-pose-estimation)

[HoreFice](https://github.com/horefice/Human-Pose-Estimation-from-RGB)

---

## My personal goals:

* I'd like to find a project I can clone.

* I'd like to find a recent project.

* I'd like to find a project with the 3d work done

* I'd like to find a project that can integrate with SteamVR. (bone locations instead of blobs/meshes)

1. state of the art 2d pose detector, this is crucial.

2. This 2d pose detector can return colored limbs corresponding to each body part or heat maps corresponding to joint, or even forward/backwards information from annoation. 

3. Time dependence ideally the network, when doing pixel to 2d pose and 2d pose to 3d map and 3d map to skeleton should take into account the previous frame, and with an internal representation of the boundary conditions of the human body pose (Perhaps a GAN?)

4. Additionally, there are other gimmicks that can be used, like Siamese network parallelism.
Where two shots of the same pose are rewarded for giving the same output.
Physics simulations of body mechanics can be used. Reprojection of 3d joints back to 2d geometery using meshes.
The 3d pose can be iteratively refined over and over again.
Sythetic data created by game data.
Additionally there should be 3d pose standarization.
Additionally 2d pose should be done well.
<!--
https://arxiv.org/pdf/1707.02439.pdf
https://arxiv.org/pdf/1803.08244.pdf
https://arxiv.org/pdf/1803.09722.pdf
https://arxiv.org/pdf/1809.07921v1.pdf
http://openaccess.thecvf.com/content_cvpr_2018/papers/Siarohin_Deformable_GANs_for_CVPR_2018_paper.pdf
https://infoscience.epfl.ch/record/256865/files/EPFL_TH8753.pdf
-->

[Domain Transfer for 3D Pose Estimation from Color Images without Manual Annotations](https://arxiv.org/pdf/1810.03707v1) (Oct 2018)

`Not relevant, but hand posing is here, so I guess it kinda is`

[Context-Aware Deep Spatio-Temporal Network for Hand Pose Estimation from Depth Images](https://arxiv.org/pdf/1810.02994v1) (Oct 2018)

`Not relvant, hand pose estimation`

:camera::package:[Cascaded Pyramid Network for 3D Human Pose Estimation Challenge](https://arxiv.org/pdf/1810.01616v1) (Oct 2018)

`Top down, seems pretty typical, nothing special goin on here`

:camera::package: [Deep Textured 3D Reconstruction of Human Bodies](https://arxiv.org/pdf/1809.06547v1.pdf) [[Project]](http://www.f-zhou.com/hpe.html) (Sep 2018)
`Not relevant. I'll make a replica out of you from soggy clay. Depth training, regular camera tests`

:camera:

[Multiview 3D human pose estimation using improved least-squares and LSTM networks](https://www.sciencedirect.com/science/article/pii/S0925231218311858) (Jul 2018) `LSTM`

[Hierarchical Contextual Refinement Networks for Human Pose Estimation](https://niexc.github.io/assets/pdf/HCRN_HPE_TIP2018.pdf) (Oct 2018)

`You start from your center and work outwards`

[Fully Automatic Multi-person Human Motion Capture for VR Applications](https://link.springer.com/chapter/10.1007/978-3-030-01790-3_3) (Sep 2018)

`$$$ Have a party, turn all your friends into skeletons then they leave`

[Propagating LSTM: 3D Pose Estimation based on Joint Interdependency](http://openaccess.thecvf.com/content_ECCV_2018/papers/Kyoungoh_Lee_Propagating_LSTM_3D_ECCV_2018_paper.pdf) (Sep 2018)

[Hockey Pose Estimation and Action Recognition using Convolutional Neural Networks to Ice Hockey][https://uwspace.uwaterloo.ca/handle/10012/13835] (Sep 2018)

[Human pose estimation method based on single depth image](http://digital-library.theiet.org/content/journals/10.1049/iet-cvi.2017.0536) (Sep 2018)

[Learning Robust Features and Latent Representations for Single View 3D Pose Estimation of Humans and Objects](https://infoscience.epfl.ch/record/256865/files/EPFL_TH8753.pdf) (Sep 2018)

[A Review of Human Pose Estimation from Single Image](https://ieeexplore.ieee.org/abstract/document/8455796) (Jul 2018)

[3D Human pose estimation on Taiji sequence](https://etda.libraries.psu.edu/files/final_submissions/17625) (Jul 2018)

`MoCap and a new biomedical dataset!`

[Human Pose Estimation Based on Deep Neural Network](https://ieeexplore.ieee.org/abstract/document/8455245) (Jul 2018)

[Multi-View CNNs for 3D Hand Pose Estimation](https://dl.acm.org/citation.cfm?id=3281721) (Jul 2018)

`Lots a people lookin at your hand`

[Multiview 3D human pose estimation using improved least-squares and LSTM networks](https://www.sciencedirect.com/science/article/pii/S0925231218311858) (Jul 2018) `LSTM`

[3-D Reconstruction of Human Body Shape from a Single Commodity Depth Camera](https://ieeexplore.ieee.org/abstract/document/8371630) (Jun 2018)

[Human Pose As Calibration Pattern; 3D Human Pose Estimation With Multiple Unsynchronized and Uncalibrated Cameras](http://openaccess.thecvf.com/content_cvpr_2018_workshops/w34/html/Takahashi_Human_Pose_As_CVPR_2018_paper.html) (Jun 2018)

[Stacked dense-hourglass networks for human pose estimation](https://www.ideals.illinois.edu/handle/2142/101155) (Apr 2018)

:door:[Simple Baselines for Human Pose Estimation and Tracking](http://openaccess.thecvf.com/content_ECCV_2018/papers/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.pdf) [[CODE]](https://github.com/Microsoft/human-pose-estimation.pytorch) (Apr 2018)

[A generalizable approach for multi-view 3D human pose regression](https://arxiv.org/pdf/1804.10462.pdf) (Apr 2018)

[A Deep Learning Based Method For 3D Human Pose Estimation From 2D Fisheye Images](https://pdfs.semanticscholar.org/8ff8/840a418f9202a33fae08997afcd2da6b19f2.pdf) (Mar 2018)

[A Unified Framework for Multi-View Multi-Class Object Pose Estimation](https://arxiv.org/pdf/1803.08103v2) (Mar 2018)

Learning Monocular 3D Human Pose Estimation from Multi-view Images (Mar 2018)

Multi-Scale Structure-Aware Network for Human Pose Estimation (Mar 2018)

Mo2Cap2: Real-time Mobile 3D Motion Capture with a Cap-mounted Fisheye Camera (Mar 2018)

[Hierarchical graphical-based human pose estimation via local multi-resolution convolutional neural network](https://aip.scitation.org/doi/full/10.1063/1.5024463) (Feb 2018)

Image-based Synthesis for Deep 3D Human Pose Estimation (Feb 2018)

:door:[LSTM Pose Machines](https://arxiv.org/pdf/1712.06316.pdf) [[CODE]](https://github.com/lawy623/LSTM_Pose_Machines) (Dec 2017)

Single-Shot Multi-Person 3D Pose Estimation From Monocular RGB (Dec 2017)

Using a single RGB frame for real time 3D hand pose estimation in the wild (Dec 2017)

:package:[Learning 3D Human Pose from Structure and Motion](http://openaccess.thecvf.com/content_ECCV_2018/papers/Rishabh_Dabral_Learning_3D_Human_ECCV_2018_paper.pdf) (Nov 2017)

:package:[Integral Human Pose Regression](https://arxiv.org/pdf/1711.08229.pdf) [[CODE]](https://github.com/JimmySuen/integral-human-pose) (Nov 2017)

[Human Pose Retrieval for Image and Video collections](http://ir.inflibnet.ac.in:8080/jspui/handle/10603/168240) (Oct 2017)
`A search engine for dancers`

:door:[Human Pose Estimation Using Global and Local Normalization](https://arxiv.org/pdf/1709.07220.pdf) (Sep 2017)

:door:[Learning Feature Pyramids for Human Pose Estimation](https://arxiv.org/pdf/1708.01101.pdf) [[CODE]](https://github.com/bearpaw/PyraNet) (Aug 2017)

:package:[Recurrent 3D Pose Sequence Machines](https://arxiv.org/pdf/1707.09695.pdf) (Jul 2017)

:door:[Self Adversarial Training for Human Pose Estimation](https://arxiv.org/pdf/1707.02439.pdf) [[CODE1]](https://github.com/dongzhuoyao/jessiechouuu-adversarial-pose)[[CODE2]](https://github.com/roytseng-tw/adversarial-pose-pytorch) (Jul 2017)

Learning Human Pose Models from Synthesized Data for Robust RGB-D Action Recognition (Jul 2017)

Faster Than Real-time Facial Alignment: A 3D Spatial Transformer Network Approach in Unconstrained Poses (Jul 2017)

[A Dual-Source Approach for 3D Human Pose Estimation from a Single Image] (https://arxiv.org/pdf/1705.02883.pdf) (May 2017)

:package:[Towards 3D Human Pose Estimation in the Wild: a Weakly-supervised Approach](https://arxiv.org/pdf/1704.02447.pdf) [[CODE]](https://github.com/xingyizhou/Pytorch-pose-hg-3d) (Apr 2017)

[Adversarial PoseNet: A Structure-Aware Convolutional Network for Human Pose Estimation](https://arxiv.org/pdf/1705.00389.pdf) (Apr 2017)

Forecasting Human Dynamics from Static Images (Apr 2017)

:package:[Compositional Human Pose Regression](https://arxiv.org/pdf/1704.00159.pdf) (Apr 2017)

2D-3D Pose Consistency-based Conditional Random Fields for 3D Human Pose Estimation (Apr 2017)

:door:[Multi-context Attention for Human Pose Estimation](https://arxiv.org/pdf/1702.07432.pdf) - [[CODE]](https://github.com/bearpaw/pose-attention) (Feb 2017)

:package:[Lifting from the Deep: Convolutional 3D Pose Estimation from a Single Image](https://arxiv.org/pdf/1701.00295.pdf) (Jan 2017)

:door:[Towards Accurate Multi-person Pose Estimation in the Wild](https://arxiv.org/pdf/1701.01779.pdf) [[CODE]](https://github.com/hackiey/keypoints) (Jan 2017)

Lifting from the Deep: Convolutional 3D Pose Estimation from a Single Image (Jan 2017)

Learning from Synthetic Humans (Jan 2017)

MonoCap: Monocular Human Motion Capture using a CNN Coupled with a Geometric Prior (Jan 2017)

:door:[RMPE: Regional Multi-person Pose Estimation](https://arxiv.org/pdf/1612.00137.pdf) [[CODE1]](https://github.com/Fang-Haoshu/RMPE)[[CODE2]](https://github.com/MVIG-SJTU/AlphaPose) (Dec 2016)

:package:[Coarse-to-Fine Volumetric Prediction for Single-Image 3D Human Pose](https://arxiv.org/pdf/1611.07828.pdf) [[CODE]](https://github.com/geopavlakos/c2f-vol-demo) (Nov 2016)

:door:[Realtime Multi-person 2D Pose Estimation Using Part Affinity Fields](https://arxiv.org/pdf/1611.08050.pdf) [[CODE]](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) (Nov 2016)

3D Human Pose Estimation from a Single Image via Distance Matrix Regression (Nov 2016)

Learning camera viewpoint using CNN to improve 3D body pose estimation (Sep 2016)

EgoCap: Egocentric Marker-less Motion Capture with Two Fisheye Cameras (Sep 2016)

:package:[Structured Prediction of 3D Human Pose with Deep Neural Networks](https://arxiv.org/pdf/1605.05180.pdf) (May 2016)

:door:[DeeperCut: A Deeper, Stronger, and Faster Multi-Person Pose Estimation Model](https://arxiv.org/pdf/1605.03170.pdf) [[CODE1]](https://github.com/eldar/deepcut-cnn)[[CODE2]](https://github.com/eldar/pose-tensorflow) (May 2016)

:door:[Recurrent Human Pose Estimation](https://arxiv.org/pdf/1605.02914.pdf) [[CODE]](https://github.com/ox-vgg/keypoint_detection) (May 2016)

Synthesizing Training Images for Boosting Human 3D Pose Estimation (Apr 2016)

Seeing Invisible Poses: Estimating 3D Body Pose from Egocentric Video - Completely insane and above the scope of science (Mar 2016)

:door:[DeepCut: Joint Subset Partition and Labeling for Multi Person Pose Estimation](https://arxiv.org/pdf/1511.06645.pdf) [[CODE]](https://github.com/eldar/deepcut) (Nov 2015)

A Dual-Source Approach for 3D Pose Estimation from a Single Image (Sep 2015)

:door:[Human Pose Estimation with Iterative Error Feedback](https://arxiv.org/pdf/1507.06550.pdf) [[CODE]](https://github.com/pulkitag/ief) (Jul 2015)

:door:[Flowing ConvNets for Human Pose Estimation in Videos](https://arxiv.org/pdf/1506.02897.pdf) [[CODE]](https://github.com/tpfister/caffe-heatmap) (Jun 2015)

:package:[3D Human Pose Estimation from Monocular Images with Deep Convolutional Neural Network](http://visal.cs.cityu.edu.hk/static/pubs/conf/accv14-3dposecnn.pdf) (Nov 2014)

:door:[Efficient Object Localization Using Convolutional Networks](https://arxiv.org/pdf/1411.4280.pdf) (Nov 2014)

:door:[MoDeep: A Deep Learning Framework Using Motion Features for Human Pose Estimation](https://arxiv.org/pdf/1409.7963.pdf) (Sep 2014)

:door:[Joint Training of a Convolutional Network and a Graphical Model for Human Pose Estimation](https://arxiv.org/pdf/1406.2984.pdf) [[CODE]](https://github.com/max-andr/joint-cnn-mrf) (Jun 2014)

:door:[Learning Human Pose Estimation Features with Convolutional Networks](https://arxiv.org/pdf/1312.7302.pdf) (Dec 2013)

:door:[DeepPose: Human Pose Estimation via Deep Neural Networks](https://arxiv.org/pdf/1312.4659.pdf) (Dec 2013)

Deep 3D Pose Dictionary: 3D Human Pose Estimation from Single RGB Image Using Deep Convolutional Neural Network
3D Hand Pose Tracking from Depth Images using Deep Reinforcement Learning

[Human 3D Reconstruction and Identification Using Kinect Sensor](https://ieeexplore.ieee.org/abstract/document/8477609) (Aug 2018) `Low-fi body ID`

[3D Head Pose Estimation Using Tensor Decomposition and Non-linear Manifold Modeling](https://ieeexplore.ieee.org/abstract/document/8491002) (Sep 2018) `

`
A Data-Driven Approach for 3D Human Body Pose Reconstruction from a Kinect Sensor
Accidental Fall Detection Based on Pose Analysis and SVDD

$$$[Global Feature Learning with Human Body Region Guided for Person Re-identification](https://link.springer.com/chapter/10.1007/978-3-030-03398-9_2) (Nov 2018)
 
XXX[HUMAN POSE ESTIMATION IN IMAGE SEQUENCES
](http://eprints.utm.my/id/eprint/79567/1/TohJunHauMFKE2018.pdf) (Jun 2018)
 
[Filling the Joints: Completion and Recovery of Incomplete 3D Human Poses](https://www.mdpi.com/2227-7080/6/4/97) (Jun 2018)
 
[Adapting MobileNets for mobile based upper body pose estimation](https://repository.edgehill.ac.uk/10789/1/adapting-mobilenets-debnath.pdf) Oct 2018
`Fast Boxer`

[3D Human Pose Estimation with 2D Marginal Heatmaps](https://arxiv.org/pdf/1806.01484v2)

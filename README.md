# InDs
This work introduces an intelligent embedding framework named InDs. It utilizes an actor-critic model based on DRL architecture to make sequential decisions in a dynamic environment considering system and network features for solving the VNE problem. InDs tries to Maximize the acceptance  and revenue-to-cost ratio and minimize congestion.


## Execution Environment
- Operation System: Microsoft Windows 10, 64bit. <br />
- Physical Memory (RAM) 16.0 GB. <br />
## Prerequisites
- Python 3.9. <br />
- PyCharm Community Edition 2021.2. <br />
- Alib utility tool for VNE simulation. <br />
## Installation
Download the ALIB_Utility tool, unzip and copy it to the execution drive. <br />
- Configure the alib by following the steps mentioned in the GitHub repository [1]. <br />
- Generate the input. pickle file and save it in the P3_ALIB_MASTER\input path. <br />
- Make sure "P3_ALIB_MASTER\input" path contain senario_RedBestel.pickle. If not, generate the substrate network scenario for "senario_RedBestel.pickle" in folder P3_ALIB_MASTER\input, and this pickle file contains substrate network information. <br />
## Download  InDs Code; It has separate zip files: one is the Training folder (all_model_train.zip), and another one is for the Testing folder (all_model_train.zip)  and Unzip both keep it in the folder  P3_ALIB_MASTER.  
- About all_model_train: This folder contains all the files responsible for training the proposed appraoch InDs and the A3C-GCN [] (a baseline appraoch).
  The input parameters required for the training and the input environment (Virtual network requests and the substrate network) settings are described in the usage part below. 

The  proposed work file contains all executable files related to the **proposed and baseline approaches**.
- InDs.py -> The Main file related to the  **Proposed** **InDs** approach. <br />
- Nrm.py -> The Main file related to the  **NRM** baseline approach [2]. <br />
- Rethinking.py -> The Main file related to the  **DPGA** baseline approach [3]. <br />
-  gcn_rl_automatic.py -> The Main file related to the  **A3C-GCN** baseline approach [4]. <br />
- greedy.py -> The Main file related to the  **VNE-MWF** baseline approach [5]. <br />


## Model Training

- Environment:- This, taken in the current state and based on the substrate node chosen, performs the embeddings.<br />
- **Step 1** :- Parameters need to be configured for training. <br />
- Number of episodes - This is the number of episodes the algorithm is getting trained. <br />

- Learning rate (lr)- It represents the learning of the neural network in each step. <br />

- Epsilon - Exploration factor, which showcases the amount of exploration required. <br />

- Workers - The number of agents running parallely. It depends on the CPU count. <br />

- Number of VNRs (vne_list)- Number of VNRs passed in each episode. <br />


- **Step 2** :-The training parameters(epsilon, number of episodes, learning rate) are specified in the respective function in automate.py and  then Run **all_model_train\automate.py** file. <br /> 

## Model Testing
 
**Note - The .pth file generated during the training is automatically  copied into the all_model_test folder.** <br />

- During testing, we can use the same substrate network and the exact virtual network requests. However, we kept the substrate network the same for better testing accuracy. Still, we generated the new set of virtual network requests by modifying the parameters in the related file described in the Usage section. <br />
- Run **test_all.py** <br />

- Check the results in the Excel file generated.<br />



## Usage

###  In vne_u.py, we can set the various parameters related to Virtual network requests(VNRs).<br />

- We can set the minimum and maximum number of VMs of VNRs in the create_vne function.<br />

- We can set the virtual network request demands like BandWidth(min, max), CRB(min, max), LocationX(min, max), LocationY(min, max), Processing_delay(min,max), Proppgation_delay(min,max). <br />
- Example: (1, 5, 1, 10, 0, 100, 0, 100,  1, 5, 1, 5)<br />

- Run vne_p.py after doing any modifications. <br />

###  In grpah_extraction_poission.py:<br />

- In the get_graphs function, mention the pickle file related to substrate network generation. The same is available in the folder P3_ALIB_MASTER. EX: os.path.join( os.path.dirname(current), "P3_ALIB_MASTER", "input", "senario_RedBestel.pickle",)<br />

- In graph.parameters function set substrate network resources like BandWidth(min,max), CRB(min,max), LocationX(min,max), LocationY(min,max), Processing_delay_server(min,max), Proppgation_delay_server(min,max)(min,max). <br />
- Example: (500, 1000, 200, 1000, 0, 100, 0, 100, 1, 5, 1, 5)<br />

- Run grpah_extraction_poisson.py after doing any modification. <br />

### grpah_p.py

- This file generates the standard 1_uniform.pickle file, which contains all the information about substrate network topologies, such as the number of servers, links, and connectivity. It also includes values for each substrate network resource. <br />

###  In the automate.py file, set the VNR size such as [250, 500, 750, 1000] and also mention the number of iterations needed to execute for each VNR size in the iteration variable.<br />

- Finally, run the test_all.py file. After successfully running, a 1_poission.pickle and 1_poisson_vne.pickle file is created related to a new set of  SN and VNRs, respectively. (If it already does not exist in the specified path). It has all input parameters related to the substrate network parameters, such as CRB and bandwidth. <br />

- Final embedding results are captured in Results.xlsx, which includes values for various metrics for all test scenarios for every iteration. <br />

### References
[1] E. D. Matthias Rost, Alexander Elvers, “Alib,” https://github.com/vnep-approx/alib, 2020. <br />
[2] P. Zhang, H. Yao, Y. Liu, Virtual network embedding based on computing, network, and storage resource constraints, IEEE Internet of Things Journal 5 (5) (2017) 3298–3304. doi: https://doi.org/10.1109/JIOT.2017.2726120. <br />
[3] Nguyen, Khoa TD, Qiao Lu, and Changcheng Huang. "Rethinking virtual link mapping in network virtualization." In 2020 IEEE 92nd Vehicular Technology Conference (VTC2020-Fall), pp. 1-5. IEEE, 2020, https://ieeexplore.ieee.org/document/9348799. <br />
[4] Yan, Zhongxia, Jingguo Ge, Yulei Wu, Liangxiong Li, and Tong Li. "Automatic virtual network embedding: A deep reinforcement learning approach with graph convolutional networks." IEEE Journal on Selected Areas in Communications 38, no. 6 (2020): 1040-1057. https://doi.org/10.1109/JSAC.2020.2986662. <br />
[5] TG, Keerthan Kumar, Ankit Srivastava, Anurag Satpathy, Sourav Kanti Addya, and S. G. Koolagudi. "MatchVNE: A Stable Virtual Network Embedding Strategy Based on Matching Theory." In 2023 15th International Conference on COMmunication Systems & NETworkS (COMSNETS), pp. 355-359. IEEE, 2023. https://doi.org/10.1109/COMSNETS56262.2023.10041377. <br />





# Full-batch Gradient Descent

ECE1747 Group 9

## Project Structure

This project consists of four programs, namely: single-threaded local execution (`./local`), distributed computing with multiple processes (`./mpi`), local GPU acceration (`./gpu`), and a multi-GPU version (`./mpigpu`).

## How to use

### Prepare the dataset

Put training data to the project folder. The training data should be named as `data.csv`. Data entries are provided in lines and each feature is seperated by comma. The last term in each row is the target value. For example, for a `m*n` dataset, there's `m` data in total and `n-1` features.

The dataset can be of any size, however, each entries must have the same amount of features.

### Compile the Program

Make sure the machine has correctly installed CUDA and MPI support.

To compile the program, one should first determine which version he/she is going to use. Go to the folder of that implementation in shell, and use command below:

```shell
cmake .
make
```

### Run the Program

To run any GPU-related implementations, make sure the GPU is correctly installed. To run any distributed memory system, be sure the machine is inside any computer cluster. To make sure the MPI will run correctly, you can try the following command:

```shell
ssh other_machines_in_cluster # e.g. ug56
```

If the ssh success, the computer is ready to be run as computation node using mpi. Keep in mind the name of the other machines in the cluster.

#### Run local version

```shell
local
```

#### Run local GPU version

```shell
gpu
```

#### Run mpi version

If we are running the ug55, and want to activate `ug55`, `ug56`, `ug57`, `ug58` as nodes

```shell
mpirun -H ug55,ug56,ug57,ug58 mpi
```

#### Run multi-gpu (mpigpu) version

If we are running the ug55, and want to activate `ug55`, `ug56`, `ug57`, `ug58` as nodes

```shell
mpirun -H ug55,ug56,ug57,ug58 mpigpu
```

### Result and evaluation

During the run, the program will output the current epoch running, parameter weight, and the MSE value of that epoch. The last output would be the optimal weight of the problem. The weight will have `n+1` values (`n` is the number of feature). The last value stand for the bias (constant term) of the gradient descent problem.

The program will also record speed of execution. The timer will be started **after the file reading is complelte**. So any time spent in communication, thread creation will also be counted as execution time. We can evaluate the performance of each version easily by comparing the execution time.

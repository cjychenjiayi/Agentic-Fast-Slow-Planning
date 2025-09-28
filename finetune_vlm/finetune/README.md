Please refer to the official guidance of **Intern-VL**
For param freeze strategy, please adjust the freeze param in the finetune shell

We use the learning rate **8e-5** with batch size **128**, the model is train on a server with 8 A100 GPU
The epoch is set to **12**, which is the **best** epoch in our scenario under the verification of **validation set**
All the experiment is done under same setting
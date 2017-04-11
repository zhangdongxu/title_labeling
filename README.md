
This tool helps you to count co-occurrence frequency of descriptors given title names.
When the service starts, it requires users' queries and returns a list of title names most related.
This code is written in Python3.

----

1. To count frequency, you can simply change the first few lines of count.sh and run the shell:

   ```shell
   nohup sh count.sh > log.logname &
   ```
   You need to modify variables in the top of this script. 
   This script read some corpus files and output a model for further usage.
   
2. To start a server in a docker container, you can use following command, the server load model.p and save log file to log.server:
   ```shell
   #build a docker image. This command will only be executed one time. 
   sudo docker build -t <image name> .

   #then, copy your model to model/model.p, there is a demo model saved there.
   cp model_path model/model.p
    
   #start a service by running a container over the image.
   sudo docker run -p <port>:5011 -v $PWD:/home/cpp/title_labeling --name <container name> -d <image name>
   ```
3. If you want to start a service with movie_server.py, then you should install grpc: 
   ```shell
   pip3 install grpcio
   ```
   
   If you want to re-design a .proto file and compile it, you should also install grpc tool:
   ```shell
   pip3 install grpcio-tools
   ```
4. You can also use *descriptor.py* for other specific needs:

   ```shell
   #help
   python3 descriptor.py --help

   #build a model with a single corpus file input
   python3 descriptor.py -b --model_type weightedwindow --window_size window_size --smooth_factor smooth_factor --input your_input_file --descriptor your_desc_file --title your_title_file  --model output_model_path
   
   #merge models you previously built
   python3 descriptor.py -m --mergedir output_model_directory --model output_model_path 

   #cleanup title names in the model.
   python3 descriptor.py -c --model input_model_path --cleaned_model output_model_path

   #prune a model with given threshold and model file
   python3 descriptor.py -p --prune_threshold 1.0 --model input_model_path --pruned_model output_pruned_model_path
   
   #load a model and look at rankings in a interactive mode. You can input a string containing several descriptions.
   python3 descriptor.py -fl --model model_path_you_want_to_load --partial_rank

   #load a model and look at rankings in a interactive mode. You can input a single description or title name and check out count details.
   python3 descriptor.py -l --model model_path_you_want_to_load 

   #load a model and generate a full ranking list given a query. Here we suggest you load a cleaned model.
   python3 descriptor.py -q --model model_path_you_want_to_load --query_string 电影 > DianYing.list 

   #load a model and evaluate it with a evaluation set
   python3 descriptor.py -e --testset data/evaluation.p --partial_rank --model input_model_path
   ```

5. You can import descriptor.py and instantiate `class Descriptor` or `class DescriptorWeightedWindow`  etc. Or you can inherit them for your needs.

   ```python
   class Descriptor:
       def count_freq(self) #virtual
       def save_model(self, model_file)
       def merge_model(self, merge_dir, output_model_file)
   class DescriptorParagraph(Descriptor)
   class DescriptorWindow(Descriptor)
   class DescriptorWeightedWindow(DescriptorWindow)
   ```
----

P.S. 

Three strategies can be implemented with parameter `--model_type`: 
* Paragraph-wise co-occurrence (line-wise)
* Window-based co-occurrence
* Weighted window-based co-occurrence, following the formula: $ smoothingfactor/(smoothingfactor + distance) $

When scoring a title given a description, we follow fomulas below:

For -q mode:

P(d, c| t) 
           
           = P(d| c, t)                 * P(c| t) 

           = P(d| c, t)                 * (P(t| c) * P(c) / P(t))

           = [freq(d, c, t)/freq(c, t)] * ((freq(c, t) / freq(c)) * P(c) / P(t))

           = C1 * freq(d, c, t) / P(t)

where for example, d is "电影", c is "《》" and t is "阿甘正传"

logP(t) = max(P(t1) * P(t2)... , -15.7357) where ti is max substring using forward maxmatch. 


For -fl, -l and server mode:

score(d, t) = 

              log(freq(d, c, t) / sqrt(freq(t, c)))


*Edited by Dongxu Zhang on April 10th, 2017.*
   ​

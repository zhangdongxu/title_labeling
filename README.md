
This tool helps you to count co-occurrence frequency of descriptors given movie titles.
When the service started, it requires users' queries and returns a list of movie titles most related.

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
   
   #start a service by running a container over the image.
   sudo docker run -p <port>:5011 -v $PWD:/home/cpp/title_labeling --name <container name> -d <image name>
   ```
3. If you want to start a service with movie_server.py, then you should install grpc: 
   ```shell
   pip install grpcio
   ```
   
   If you want to re-design a .proto file and compile it, you should also install grpc tool:
   ```shell
   pip install grpcio-tools
   ```
4. You can also use *descriptor.py* for other specific needs:

   ```shell
   #help
   python descriptor.py --help

   #build a model with a single corpus file input
   python descriptor.py -b --model_type weightedwindow --window_size window_size --smooth_factor smooth_factor --input your_input_file --descriptor your_desc_file --title your_title_file  --model output_model_path
   
   #prune a model with given threshold and model file
   python descriptor.py -p --prune_threshold 1.0 --model input_model_path --prune_file output_pruned_model_path
   
   #merge models you previously built
   python descriptor.py -m --mergedir output_model_directory --model output_model_path 

   #load a model and look at rankings in a interactive mode
   python descriptor.py -fl --model model_path_you_want_to_load --rank_method and --partial_rank
   python descriptor.py -l --model model_path_you_want_to_load 

   #load a model and evaluate it with a evaluation set and a ranking method
   python descriptor.py -e --testset evaluation.p --rank_method and --partial_rank --model input_model_path
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

*Edited by Dongxu Zhang on Feb 24th, 2017.*
   ​

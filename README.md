
This tool helps you to count co-occurrence frequency of descriptors given movie titles.
If you want to start a service with movie_server.py, then you should install grpc: 
```shell
pip install grpcio
pip install grpcio-tools
```


----
Three strategies can be implemented with parameter $model_type$ : 
* Paragraph-wise co-occurrence (line-wise)
* Window-based co-occurrence
* Weighted window-based co-occurrence, following the formula: $ smoothingfactor/(smoothingfactor + distance) $

input files:  titleFile descriptorFile inputFiles

----

1. To count frequency, you can simply change the first few lines of count.sh and run the shell:

   ```shell
   nohup sh count.sh > log.logname &
   ```
2. To start a server, you can use following command:
   ```shell
   nohup python movie_server.py --model model_path_you_want_to_load > log.server.model_name &
   ```

3. You can also use *descriptor.py* for other specific needs:

   ```shell
   #help
   python descriptor.py -h

   #build a model with a single corpus file input
   python descriptor.py -b --model_type weightedwindow --window_size window_size --smooth_factor smooth_factor --input your_input_file --descriptor your_desc_file --title your_title_file  --model output_model_path

   #merge models you previously built
   python descriptor.py -m --mergedir output_model_directory --model output_model_path 

   #load a model and look at rankings in a interactive mode
   python descriptor.py -fl --model model_path_you_want_to_load 
   python descriptor.py -l --model model_path_you_want_to_load 
   ```

4. You can import descriptor.py and instantiate `class Descriptor` or `class DescriptorWeightedWindow`  etc. Or you can inherit them for your needs.

   ```python
   class Descriptor:
       def count_freq(self) #virtual
       def save_model(self, model_file)
       def merge_model(self, merge_dir, output_model_file)
   class DescriptorParagraph(Descriptor)
   class DescriptorWindow(Descriptor)
   class DescriptorWeightedWindow(DescriptorWindow)
   ```

*Edited by Dongxu Zhang on Feb 24th, 2017.*
   ​

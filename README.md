# auto-learn-model
Based on concept of genetic algorithm to automatically learn the super parameters of the learning model, such as LR, RF, GBDT, and so on

## Usage:
Many configurations could be found within TaskBuilder   <br>
* `[Required]`    Modify TaskBuilder.loadData function, to load the trainning data, within which the target column should be titled with "label".   <br>
* `[Required]`    Modify TaskBuilder.getConvergenceRecordOutputPath function, to specify the output path of the records which logged the trace of converged state.    <br>
* `[Required]`    Modify TaskBuilder.getBestResultsOutputPath function, to specify the output path of the best probing results.   <br>
* `[Optional]`    Modify TaskBuilder.loadOperators function, to load suitable operators, or even you could define your own operator within operators package, and then load it.   <br>
* `[Optional]`    Modify TaskBuilder.getBeamSearchNum function, to specify the number of probers, which depend on your calculating resources. <br>
* `[Optional]`    Modify TaskBuilder.initContext function, to initailize sparkSession, e.g., you could specify the running mode of your calculating platform here.  <br>

With the essential comprehension about other parameters and functions, you could change them for your own interests, or even just test.  <br>

Run MasterConsole to start probing, you can terminate the program till the BestResultsOutputPath has output, and in general it needs about 20 minutes in case of current configurations,
and of the cluster mode it will be faster. But more times you are waiting, much better result would be found. Of course, the program would stop by itself, that depend on its converging state.   <br>








# auto-learn-model
基于遗传算法框架，自动学习各种学习模型的超参数，比如LR, RF, GBDT等等

## 使用方法：
任务的配置大都在TaskBuilder中    <br>
* 必需更改TaskBuilder的loadData方法，以加载训练文件，训练文件的target列需要命名为“label”    <br>
* 必需更改TaskBuilder的getConvergenceRecordOutputPath方法，来指定收敛记录输出路径  <br>
* 必需更改TaskBuilder的getBestResultsOutputPath方法，来指定最好结果的输出路径   <br>
* 可以更改TaskBuilder的loadOperators方法，来加载别的算子，或者可以在operators包下添加你自己的算子，然后再加载进来  <br>
* 可以更改TaskBuilder的getBeamSearchNum方法，来指定同时有多少个agent在执行搜索任务，当然最好的方法是根据当前集群（单机）的计算资源来计算后确定    <br>
* 可以更改TaskBuilder的initContext方法，来指定运行模式（local、cluster等），初始化sparkSession   <br>

至于其余的配置，建议了解参数或方法的意义和可能产生的影响后进行修改   <br>

运行MasterConsole开始执行任务，当最好结果输出路径下有输出时，可以随时停止，以现在的单机配置一般需要等20分钟左右，集群模式下会更快。
当然运行时间越长，越有可能得到更好的结果。系统本身有个收敛判断，届时程序会自动停止。   <br>

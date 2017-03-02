# auto-learn-model
Based on concept of genetic algorithm to automatically learn the super parameters of the learning model, such as LR, RF, GBDT, and so on

##Usage:
Many configurations could be found within TaskBuilder   <br>
1.[Required]    Modify TaskBuilder.loadData function, to load the trainning data.   <br>
2.[Required]    Modify TaskBuilder.getConvergenceRecordOutputPath function, to specify the output path of the records which logged the trace of converged state.    <br>
3.[Required]    Modify TaskBuilder.getBestResultsOutputPath function, to specify the output path of the best probing results.   <br>
4.[Optional]    Modify TaskBuilder.loadOperators function, to load suitable operators, or even you could define your own operator within operators package, and then load it.   <br>
5.[Optional]    Modify TaskBuilder.getBeamSearchNum function, to specify the number of probers, which depend on your calculating resources. <br>
6.[Optional]    Modify MasterConsole.initContext function, to initailize sparkSession, e.g., you could specify the running mode of your calculating platform here.  <br>
With the essential comprehension about other parameters and functions, you could change them for your own interests, or even just testing.  <br>








# auto-learn-model
基于遗传算法框架，自动学习各种学习模型的超参数，比如LR, RF, GBDT等等

##使用方法：
任务的配置大都在TaskBuilder中    <br>
1、必需更改TaskBuilder的loadData方法，以加载训练文件    <br>
2、必需更改TaskBuilder的getConvergenceRecordOutputPath方法，来指定收敛记录输出路径  <br>
3、必需更改TaskBuilder的getBestResultsOutputPath方法，来指定最好结果的输出路径   <br>
4、可以更改TaskBuilder的loadOperators方法，来加载别的算子，或者可以在operators包下添加你自己的算子，然后再加载进来  <br>
5、可以更改TaskBuilder的getBeamSearchNum方法，来指定同时有多少个agent在执行搜索任务，当然最好的方法是根据当前集群（单机）的计算资源来计算后确定    <br>
6、可以更改MasterConsole的initContext方法，来指定运行模式（local、cluster等），初始化sparkSession   <br>
至于其余的配置，建议了解参数或方法的意义和可能产生的影响后进行修改   <br>
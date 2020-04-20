# Finding GENERAL Defect Prediction Models Within Hundreds of Software Projects

```
        _       _       _       _       _       _       _       _
     _-(_)-  _-(_)-  _-(_)-  _-(_)-  _-(_)-  _-(_)-  _-(_)-  _-(_)-
   `(___)  `(___)  `(___)  `(___)  `(___)  `(___)  `(___)  `(___)
    // \\   // \\   // \\   // \\   // \\   // \\   // \\   // \\
        _       _       _       _       _       _       _       _
     _-(_)-  _-(_)-  _-(_)-  _-(_)-  _-(_)-  _-(_)-  _-(_)-  _-(_)-
   `(___)  `(___)  `(___)  `(___)  `(___)  `(___)  `(___)  `(___)
    // \\   // \\   // \\   // \\   // \\   // \\   // \\   // \\
       _       _       _       _       _       _       _       _
    _-(_)-  _-(_)-  _-(_)-  _-(")-  _-(_)-  _-(_)-  _-(_)-  _-(_)-
  `(___)  `(___)  `(___)  `%%%%%  `(___)  `(___)  `(___)  `(___)
   // \\   // \\   // \\   // \\   // \\   // \\   // \\   // \\
       _       _       _       _       _       _       _       _
    _-(_)-  _-(_)-  _-(_)-  _-(_)-  _-(_)-  _-(_)-  _-(_)-  _-(_)-
  `(___)  `(___)  `(___)  `(___)  `(___)  `(___)  `(___)  `(___)
   // \\   // \\   // \\   // \\   // \\   // \\   // \\   // \\
       _       _       _       _       _       _       _       _
    _-(_)-  _-(_)-  _-(_)-  _-(_)-  _-(_)-  _-(_)-  _-(_)-  _-(_)-
  `(___)  `(___)  `(___)  `(___)  `(___)  `(___)  `(___)  `(___)
   // \\   // \\   // \\   // \\   // \\   // \\   // \\   // \\
```
## Abstract
Given a large group of software projects, there often exists  a few   exemplar project(s) that offers the best prediction for all others. Such ''bellwether projects'' can be used to make quality predictions that are general to many other projects.

Existing  methods  for  finding bellwether  have two problems. Firstly, they are  very slow.  When  applied  to  the  697  projects  studied  here, standard bellwether methods   took  60  days  of  CPU  to  find  and  certify  the  bellwethers.
Secondly, they assume that only one bellwether exists and, amongst hundreds of projects, they may exist  subgroups, each of which requires their own bellwether.  

GENERAL is a new bellwether method that addresses both problems. GENERAL applies hierarchical clustering to groups of project data. At each level within a tree of clusters, one bellwether is computed from sibling projects, then promoted up the tree.  In this way, GENERAL can find multiple
bellwethers, if they exist. Also, GENERAL's hierarchical decomposition runs much faster (and scales better) than standard
bellwether methods.  


This site contains the scripts and data needed to support:

```
@misc{majumder2019learning,
    title={Learning GENERAL Principles from Hundreds of Software Projects},
    author={Suvodeep Majumder and Rahul Krishna and Tim Menzies},
    year={2019},
    eprint={1911.04250},
    archivePrefix={arXiv},
    primaryClass={cs.SE}
}
```

To reproduce the results follow the instruction for each research question.
## RQ1: How slow are conventional bellwethermethods?
This code will run the traditional bellwether at level 0 showing a N^2 comparison. The code will report the runtime for a 10 * 2 comparison.
command: sh RQ1.sh

## RQ2: In theory, how much faster is GENERAL's hierarchical clustering?
This code will run the new bellwether methos aka GENERAL. The code will report the runtime for a 10 * 2 comparison.
command: sh RQ2.sh

## RQ3: In practice, how fast is  GENERAL?
This code will run the traditional bellwether and new bellwether methos aka GENERAL with various community size. The code will report the runtime for a 10 * 2 comparison.
command: sh RQ3.sh

## RQ4: Is this faster bellwether effective?
This code will run take the bellwethers found by traditional bellwether, GENERAL at different levels along with other models used in this study and predict its performance on a test dataset.
command: sh RQ4.sh

## RQ5: Does learning from too many projects have detrimental effects?
This code will run the traditional bellwether at level 0 showing a N^2 comparison. The code will report the runtime for a 10 * 2 comparison.
command: sh RQ5.sh


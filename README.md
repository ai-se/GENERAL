# Finding GENERAL Defect Prediction ModelsWithin Hundreds of Software Projects

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

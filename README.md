# Solutions of Reinforcement Learning 2nd Edition ([Original Book](https://www.amazon.co.jp/exec/obidos/ASIN/0262039249/hatena-blog-22/) by Richard S. Sutton,Andrew G. Barto)

## How to contribute and current situation (9/11/2021~)

I have been working as a full-time AI engineer and barely have free time to manage this project any more. I want to make a simple guidance of how to response to contributions:

### For exercises that have no answer yet, (for example, chapter 12)
1. Prepare your latex code, make sure it works and looks somewhat nice.
2. Send you code to ericwang.usa@gmail.com. By default, I will put contributer's name in the pdf file, besides the exercise. You can be anoymous as well just noted in the email.
3. I will update the corresponding solution pdf.

### For solution that you think is wrong, but it is trivial to change:
1. Ask in issues. If there are multiple confirmations and reports to the same issue, I will change the excercise. (the pass rate of such issue is around 30%)

###  For solution that you think is wrong or incomplete, but it is hard to say that in issue. 
Follow the first steps (just as if this exercise has no solution)

I know there is an automatic-ish commit and contribute to pdf procedure, but from the number of contributions, I decide to pass it on. (currently only 2% is contributed by person other than me)

Now I am more concentrated on computer vision and have less time contributing to the interest (RL). But I do hope and think RL is the future subject that will be on the top of AI pyramid one day and I will come back. Thanks for all your supports and best wishes to your own careers.


### Those students who are using this to complete your homework, stop it. This is written for serving millions of self-learners who do not have official guide or proper learning environment. And, Of Course, as a personal project, it has ERRORS. (Contribute to issues if you find any).

Welcome to this project. It is a tiny project where we don't do too much coding (yet) but we cooperate together to finish some tricky exercises from famous RL book Reinforcement Learning, An Introduction by Sutton. You may know that this book, especially the second version which was published last year, has no official solution manual. If you send your answer to the email address that the author leaved, you will be returned a fake answer sheet that is incomplete and old. So, why don't we write our own? Most of problems are mathematical proof in which one can learn the therotical backbone nicely but some of them are quite challenging coding problems. Both of them will be updated gradually but math will go first.

Main author would be me and current main cooperater is **Jean Wissam Dupin**, and before was Zhiqi Pan (quitted now). 

### Main Contributers for Error Fixing:

### Simon Haastert's work
Chapter 9: 9.6

### burmecia's Work (Error Fix and code contribution)
Chapter 3: 
Ex 3.4, 3.5, 3.6, 3.9, 3.19

Chapter4:
Ex 4.7 Code(in Julia)

### Jean's Work (Error Fix):

Chapter 3:
Ex 3.8, 3.11, 3.14, 3.23, 3.24, 3.26, 3.28, 3.29, 4.5

### QihuaZhong's Work (Error fix, analysis)

Ex 6.11, 5.11, 10.5, 10.6

### luigift's Work (Error fix, algorithm contribution)
Ex 10.4 10.6 10.7
Ex 12.1 (alternative solution)

### Other people (Error Fix):

Ex 10.2  SHITIANYU-hue 
Ex 10.6 10.7 Mohammad Salehi 

### ABOUT MISTAKES:

Don't even expect the solutions be perfect, there are always mistakes. Especially in Chapter 3, where my mind was in a rush there. And, sometimes the problems are just open. Show your ideas and question them in 'issues' at any time!


Let's roll'n out!

### UPDATE LOG:

**Will update and revise this repo after 2021 April**

[UPDATE APRIL 2020] After implementing Ape-X and D4PG in my another project, I will go back to this project and at least finish the policy gradient chapter. 

[UPDATE MAR 2020] Chapter 12 almost finished and is updated, except for the last 2 questions. One for dutch trace and one for double expected SARSA. They are tricker than other exercises and I will update them little bit later. Please share your ideas by opening issues if you already hold a valid solution.**

[UPDATE MAR 2020] Due to multiple interviews ( it is interview season in japan ( despite the **virus**!)), I have to postpone the plan of update to March or later, depending how far I could go. (That means I am doing leetcode-ish stuff every day)

[UPDATE JAN 2020] Future works will NOT be stopped. I will try to finish it in FEB 2020.

[UPDATE JAN 2020] Chapter 12's ideas are not so hard but questions are very difficult. (most chanllenging one in this book
). As far, I have finished up to Ex 12.5 and I think my answer of Ex 12.1 is the only valid one on the internet (or not, challenge welcomed!) But because later half is even more challenging (tedious when it is related to many infiite sums), I would release the final version little bit later.

[UPDATE JAN 2020] Chapter 11 updated. One might have to read the referenced link to Sutton's paper in order to understand some part. Espeically how and why Emphatic-TD works.

[UPDATE JAN 2020] Chapter 10 is long but interesting! Move on!

[UPDATE DEC 2019] Chapter 9 takes long time to read thoroughly but practices are surprisingly just a few. So after uploading the Chapter 9 pdf and I really do think I should go back to previous chapters to complete those programming practices.

# Chapter 12

[Updated March 27] Almost finished. 

[CHAPTER 12 SOLUTION PDF HERE](https://github.com/LyWangPX/Solutions-of-Reinforcement-Learning-An-Introduction-Sutton-2nd/blob/master/Chapter%2012/Solutions_to_Reinforcement_Learning_by_Sutton_Chapter_12_rx.pdf)

# Chapter 11
Major challenges about off-policy learning. Like Chapter 9, practices are short.

[CHAPTER 11 SOLUTION PDF HERE](https://github.com/LyWangPX/Solutions-of-Reinforcement-Learning-An-Introduction-Sutton-2nd/blob/master/Chapter%2011/Solutions_to_Reinforcement_Learning_by_Sutton_Chapter_11_r2.pdf)

# Chapter 10
It is a substantial complement to Chapter 9. Still many open problems which are very interesting.

[CHAPTER 10 SOLUTION PDF HERE](https://github.com/LyWangPX/Solutions-of-Reinforcement-Learning-An-Introduction-Sutton-2nd/blob/master/Chapter%2010/Solutions_to_Reinforcement_Learning_by_Sutton_Chapter_10_r7.pdf)

# Chapter 9
Long chapter, short practices.

[CHAPTER 9 SOLUTION PDF HERE](https://github.com/LyWangPX/Solutions-of-Reinforcement-Learning-An-Introduction-Sutton-2nd/blob/master/Chapter%209/Solutions_to_Reinforcement_Learning_by_Sutton_Chapter_9.pdf)

# Chapter 8
Finished without programming. Plan on creating additional exercises to this Chapter because many materials are lack of practice.

[CHAPTER 8 SOLUTION PDF HERE](https://github.com/LyWangPX/Solutions-of-Reinforcement-Learning-An-Introduction-Sutton-2nd/blob/master/Chapter%208/Solutions_to_Reinforcement_Learning_by_Sutton_Chapter_8_rx.pdf)

# Chapter 7
Finished without programming. Thanks for help from Zhiqi Pan.

[CHAPTER 7 SOLUTION PDF HERE](https://github.com/LyWangPX/Solutions-of-Reinforcement-Learning-An-Introduction-Sutton-2nd/blob/master/Chapter%207/Solutions_to_Reinforcement_Learning_by_Sutton_Chapter_7_r2.pdf)

# Chapter 6
Fully finished.

[CHAPTER 6 SOLUTION PDF HERE](https://github.com/LyWangPX/Solutions-of-Reinforcement-Learning-An-Introduction-Sutton-2nd/blob/master/Chapter%206/Solutions_to_Reinforcement_Learning_by_Sutton_Chapter_6_rx.pdf)

# Chapter 5
Partially finished.

[CHAPTER 5 SOLUTION PDF HERE](https://github.com/LyWangPX/Solutions-of-Reinforcement-Learning-An-Introduction-Sutton-2nd/blob/master/Chapter%205/Solutions_to_Reinforcement_Learning_by_Sutton_Chapter_5_r3.pdf)

# Chapter 4
Finished. 
Ex4.7 Partially finished. 
Dat DP question will burn my mind and macbook but I encourage any one who cares nothing about that trying to do yourself. Running through it forces you remember everything behind ordinary DP.:)

[CHAPTER 4 SOLUTION PDF HERE](https://github.com/LyWangPX/Solutions-of-Reinforcement-Learning-An-Introduction-Sutton-2nd/blob/master/Chapter%204/Solutions_to_Reinforcement_Learning_by_Sutton_Chapter_4_r5.pdf)

# Chapter 3 (I was in a rush in this chapter. Be aware about strange answers if any.)

[CHAPTER 3 SOLUTION PDF HERE](https://github.com/LyWangPX/Solutions-of-Reinforcement-Learning-An-Introduction-Sutton-2nd/blob/master/Chapter%203/Solutions_to_Reinforcement_Learning_by_Sutton_Chapter_3_rx1.pdf)


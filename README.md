Before using optuna create a super-short(like 15 frames) yuv video to pass it, use sample_yuv_frames function from tools.py
Also don't forget to pass the same output file in optuna(not passing any should work and save every filtered video in out.yuv  
But passing the same output file will cause inability to set n_jobs != 1 
  
files look like this:  
  
Your folder  
.spatter.py  
.tools.py  
.abstract.py  
.videos  
..crowd_run_short_1920x1080_50.yuv  
..the rest of the videos  

  link to mp4 filtered files: https://disk.yandex.com/d/d6FLdJa00PVgag

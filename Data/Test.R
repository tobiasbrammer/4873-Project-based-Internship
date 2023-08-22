# put this before start of loop
    total <- 10
    # put this before closing braces of loop
    pb <- winProgressBar(title = "progress bar", min = 0, max =total , width = 300)
      Sys.sleep(0.1)
    # Here i is loop itrator
      setWinProgressBar(pb, i, title=paste( round(i/total*100, 0),"% done"))
    # put this after closing braces of loop
    close(pb)
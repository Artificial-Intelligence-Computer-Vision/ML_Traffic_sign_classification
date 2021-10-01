import splitfolders

splitfolders.ratio("traffic_signs/Train", output="traffic_signs/Train_split_25", seed=1337, ratio=(.50, .25, .25), group_prefix=None)

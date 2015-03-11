require(ggplot2)
require(plyr)

data <- read.csv("./benchmark_local_8.csv", sep=";", header=TRUE)

cdata <- ddply(data, c("p", "method"), summarise, time=mean(time))

fig <- ggplot(cdata, aes(x=method, y=time)) + geom_bar()
plot(fig)

require(ggplot2)
require(plyr)

#data <- read.csv("./benchmark_local_8.csv", sep=";", header=TRUE)
data <- read.csv("./cyence_methods.csv", sep=";", header=TRUE)

cdata <- ddply(data, c("p", "method"), summarise, time=mean(time)/1000)

fig <- ggplot(cdata, aes(x=p, y=time, group=method, lty=method, color=method)) +
        geom_line() + ylab("Time [s]")
plot(fig)

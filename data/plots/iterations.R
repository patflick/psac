
#require(reshape2)
#require(ggplot2)

data <- read.csv("../size_time_iterations.tsv", sep="\t", header=TRUE)

#d <- melt(data, id=c("i"))

# for PDF output:
pdf("iterations.pdf", width=6.2, height=4.4)


par(mar=c(5,4,4,5)+.1)
barplot(data$n, axes=FALSE, ylim=c(0,3500000000))
axis(4, 1000000000*c(0, 1, 2, 3), c("0", "1 G", "2 G" , "3 G"))
mtext("Non-singleton elements", side=4, line=3)
par(new=TRUE)
plot(data$i, data$algo2/1000,
    xlim=c(-.4,16.4), ylim=c(0,max(data$algo2, na.rm=TRUE)*1.1/1000),
    type="o", lty=2, pch=17, yaxs="i",
    xlab="Iteration", ylab="Runtime per iteration [s]",
    main="Runtime per Iteration for Human Genome")
lines(data$i, data$algo1/1000, type="o", lty=1, pch=15)
legend("topright",
       c("Number Non-singleton elements", "Runtime Algorithm 1", "Runtime Algorithm 2"),
       fill=c("grey70", "white", "white"),lty=c(NA, 1, 2),
       border=c("black", "white", "white"), pch=c(NA,15,17))

# write PDF
dev.off()

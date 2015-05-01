
#require(reshape2)
#require(ggplot2)

data <- read.csv("../size_time_iterations_k21.tsv", sep="\t", header=TRUE)

#d <- melt(data, id=c("i"))

# for PDF output:
pdf("iterations_k21.pdf", width=6.2, height=4.4)


par(mar=c(5,4,4,5)+.1)
barplot(data$n, axes=FALSE, ylim=c(0,3500000000))
axis(4, 1000000000*c(0, 1, 2, 3), c("0", "1 G", "2 G" , "3 G"))
mtext("Non-singleton elements", side=4, line=3)
par(new=TRUE)
plot(data$i, data$time/1000,
    xlim=c(-.4,12.4), ylim=c(0,6),
    type="o", lty=1, pch=16, yaxs="i",
    xlab="Iteration", ylab="Runtime per iteration [s]",
    main="Runtime per Iteration for Human Genome (k=21)")
#lines(data$i, data$algo1/1000, type="o", lty=1, pch=15)
legend("topright",
       c("Number Non-singleton elements", "Runtime Algorithm 1+2"),
       fill=c("grey70", "white"),lty=c(NA, 1),
       border=c("black", "white"), pch=c(NA,16))

# write PDF
dev.off()

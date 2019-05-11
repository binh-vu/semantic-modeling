FROM ubuntu:14.04

RUN apt-get update && \
	apt-get install -y wget software-properties-common git && \
	add-apt-repository ppa:webupd8team/java && apt-get update && \
	echo debconf shared/accepted-oracle-license-v1-1 select true | sudo debconf-set-selections && \
	echo debconf shared/accepted-oracle-license-v1-1 seen true | sudo debconf-set-selections && \
	apt-get install -y oracle-java8-installer

RUN	wget http://mirrors.koehn.com/apache/maven/maven-3/3.3.9/binaries/apache-maven-3.3.9-bin.tar.gz && \
	tar -xf apache-maven-3.3.9-bin.tar.gz && \
	rm apache-maven-3.3.9-bin.tar.gz

ENV PATH /apache-maven-3.3.9/bin:$PATH
ENV JAVA_HOME /usr/lib/jvm/java-8-oracle

RUN git clone https://github.com//Web-Karma.git && \
    cd Web-Karma && \
    git checkout time_traveler && \
    mvn clean install -Dmaven.test.skip=true && \
    cd karma-research && \
    mvn compile install assembly:single && \
    mv target/karma-research-0.0.1-SNAPSHOT-jar-with-dependencies.jar /karma-research.jar

# max perm size 128m was remove in java 8
ENV MAVEN_OPTS "-Xmx4096m"

ENTRYPOINT ["java", "-jar", "karma-research.jar"]

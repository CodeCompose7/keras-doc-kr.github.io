# 베이스 이미지를 설정합니다. 이 예제에서는 Ruby 3.1을 사용합니다.
FROM ruby:3.1

# 필요한 패키지를 설치합니다.
RUN apt-get update -qq && apt-get install -y build-essential

# 작업 디렉토리를 설정합니다.
WORKDIR /usr/src/app

# Gemfile과 Gemfile.lock을 복사합니다.
COPY Gemfile Gemfile.lock ./

# Bundler 버전을 설정합니다.
RUN gem install bundler -v 2.3.26

# Gemfile에 명시된 gem을 설치합니다.
RUN bundle install

# Jekyll을 빌드하고 서버를 시작하는 명령어를 실행합니다.
CMD ["bundle", "exec", "jekyll", "serve", "--host", "0.0.0.0", "--baseurl", ""]

# 컨테이너가 0.0.0.0:4000에서 실행되도록 합니다.
EXPOSE 4000

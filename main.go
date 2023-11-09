package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"time"

	tokenizer "github.com/tiktoken-go/tokenizer"

	gpt3 "github.com/PullRequestInc/go-gpt3"
	"github.com/joho/godotenv"
)

const (
	inputFileName  = "input_code.txt"
	outputFileName = "output_code.txt"
	MaxTokensLimit = 3500
)

func main() {
	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file:", err)
	}

	apiKey := os.Getenv("CHATGPT_API_KEY")
	if apiKey == "" {
		log.Fatal("ChatGPT API key is missing")
	}

	client := gpt3.NewClient(apiKey)

	inputFile, err := os.ReadFile(inputFileName)
	if err != nil {
		log.Fatal("Error reading input code file:", err)
	}

	prompt := "Given this code, check if context is passed as a parameter. If it's missing, add context as a parameter and provide the correct code:\n\n"
	inputMsg := prompt + string(inputFile)

	tokenCount, err := countTokens(inputMsg, client)
	if err != nil {
		log.Fatal("Error counting tokens:", err)
	}
	if tokenCount > MaxTokensLimit {
		log.Fatalf("Input exceeds the maximum token limit. Actual tokens: %d, Maximum tokens: %d", tokenCount, MaxTokensLimit)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	outputFile, err := os.Create(outputFileName)
	if err != nil {
		log.Fatal("Error creating output code file:", err)
	}
	defer outputFile.Close()

	err = makeAPICall(ctx, client, inputMsg, outputFile)
	if err != nil {
		log.Fatal("Error calling ChatGPT API:", err)
	}
	fmt.Printf("Code generation completed successfully. Output saved to %s\n", outputFileName)
}

func makeAPICall(ctx context.Context, client gpt3.Client, inputMsg string, outputFile *os.File) error {
	err := client.CompletionStreamWithEngine(ctx, gpt3.TextDavinci003Engine, gpt3.CompletionRequest{
		Prompt:      []string{inputMsg},
		Temperature: gpt3.Float32Ptr(0),
		MaxTokens:   gpt3.IntPtr(MaxTokensLimit),
		N:           gpt3.IntPtr(1),
		Echo:        false,
	}, func(resp *gpt3.CompletionResponse) {

		_, err := io.WriteString(outputFile, resp.Choices[0].Text)
		if err != nil {
			log.Fatal("Error writing to output code file:", err)
		}

	})
	if err != nil {
		log.Fatal("Error calling ChatGPT API:", err)
		return err
	}
	return nil
}

func countTokens(text string, client gpt3.Client) (int, error) {
	enc, err := tokenizer.Get(tokenizer.Cl100kBase)
	if err != nil {
		return 0, err
	}

	_, tokens, err := enc.Encode(text)
	if err != nil {
		return 0, err
	}

	return len(tokens), nil
}

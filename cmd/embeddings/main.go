package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"log/slog"
	"os"
	"strings"

	"github.com/sfomuseum/go-mobileclip"
)

func usage() {
	fmt.Fprintf(os.Stderr, "Command-line tool for deriving text or image embeddings from a MobileCLIP \"service\". Results are written as a JSON-encoded string to STDOUT.\n")
	fmt.Fprintf(os.Stderr, "Usage:\n\t%s [options] text|image arg(N) arg(N)\n\n", os.Args[0])
	fmt.Fprintf(os.Stderr, "If the first argument is \"image\" then the body of the image to derive embeddings for will be read from the second argument.\n\n")
	fmt.Fprintf(os.Stderr, "If the first argument is \"text\" then the body of the text to derive embeddings for will be determined as follows: If there are only two arguments and the second argument is \"-\" then the body will be read from STDIN. Otherwise, if there are only two arguments the body of the text will be read from the file specified in the second argument. Finally, if there are more than two arguments then the body of the text will be the concatenation of the second to the last argument.\n\n")
	fmt.Fprintf(os.Stderr, "Valid options are:\n")
	flag.PrintDefaults()
}

func main() {

	var client_uri string
	var model string

	flag.StringVar(&client_uri, "client-uri", "grpc://localhost:8080", "A valid sfomuseum/go-mobileclip.EmbeddingsClient URI.")
	flag.StringVar(&model, "model", "s0", "The name of the MobileCLIP model to use to derive embeddings. Valid options are: s0, s1, s2, blt")

	flag.Usage = usage
	flag.Parse()

	ctx := context.Background()

	cl, err := mobileclip.NewEmbeddingsClient(ctx, client_uri)

	if err != nil {
		log.Fatalf("Failed to create new embeddings client, %v", err)
	}

	args := flag.Args()

	switch len(args) {
	case 0:
		slog.Warn("Insufficient arguments")
		usage()
		return
	case 1:
		slog.Warn("Insufficient arguments")
		usage()
		return
	}

	var emb *mobileclip.Embeddings

	switch args[0] {
	case "text":

		var body []byte

		switch len(args) {
		case 2:

			switch args[1] {
			case "-":

				b, err := io.ReadAll(os.Stdin)

				if err != nil {
					log.Fatalf("Failed to read STDIN, %v", err)
				}

				body = b
			default:

				b, err := os.ReadFile(args[1])

				if err != nil {
					log.Fatalf("Failed to read file, %v", err)
				}

				body = b
			}

		default:
			body = []byte(strings.Join(args[1:], " "))
		}

		req := &mobileclip.EmbeddingsRequest{
			Model: model,
			Body:  body,
		}

		emb, err = cl.ComputeTextEmbeddings(ctx, req)

		if err != nil {
			log.Fatalf("Failed to compute embeddings, %v", err)
		}

	case "image":

		body, err := os.ReadFile(args[1])

		if err != nil {
			log.Fatalf("Failed to read file, %v", err)
		}

		req := &mobileclip.EmbeddingsRequest{
			Id:    args[1],
			Model: model,
			Body:  body,
		}

		emb, err = cl.ComputeImageEmbeddings(ctx, req)

		if err != nil {
			log.Fatalf("Failed to compute embeddings, %v", err)
		}
	}

	enc := json.NewEncoder(os.Stdout)
	err = enc.Encode(emb)

	if err != nil {
		log.Fatalf("Failed to encode embeddings, %v", err)
	}
}

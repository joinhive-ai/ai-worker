// Package main provides a small example on how to run the 'text-to-image' pipeline using the AI worker package.
package main

import (
	"context"
	"flag"
	"log/slog"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"time"

	"github.com/livepeer/ai-worker/worker"
)

func main() {
	aiModelsDir := flag.String("aiModelsDir", "runner/models", "path to the models directory")
	flag.Parse()

	containerName := "text-to-image"
	baseOutputPath := "output"

	containerImageID := "livepeer/ai-runner:latest"
	gpus := []string{"0"}

	modelsDir, err := filepath.Abs(*aiModelsDir)
	if err != nil {
		slog.Error("Error getting absolute path for 'aiModelsDir'", slog.String("error", err.Error()))
		return
	}

	modelID := "stabilityai/sd-turbo"

	w, err := worker.NewWorker(containerImageID, gpus, modelsDir)
	if err != nil {
		slog.Error("Error creating worker", slog.String("error", err.Error()))
		return
	}

	slog.Info("Warming container")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if err := w.Warm(ctx, containerName, modelID, worker.RunnerEndpoint{}, worker.OptimizationFlags{}); err != nil {
		slog.Error("Error warming container", slog.String("error", err.Error()))
		return
	}

	slog.Info("Warm container is up")

	args := os.Args[1:]
	runs, err := strconv.Atoi(args[0])
	if err != nil {
		slog.Error("Invalid runs arg", slog.String("error", err.Error()))
		return
	}

	prompt := args[1]

	req := worker.GenTextToImageJSONRequestBody{
		ModelId: &modelID,
		Prompt:  prompt,
	}

	for i := 0; i < runs; i++ {
		slog.Info("Running text-to-image", slog.Int("num", i))

		resp, err := w.TextToImage(ctx, req)
		if err != nil {
			slog.Error("Error running text-to-image", slog.String("error", err.Error()))
			return
		}

		for j, media := range resp.Images {
			outputPath := path.Join(baseOutputPath, strconv.Itoa(i)+"_"+strconv.Itoa(j)+".png")
			if err := worker.SaveImageB64DataUrl(media.Url, outputPath); err != nil {
				slog.Error("Error saving b64 data url as image", slog.String("error", err.Error()))
				return
			}

			slog.Info("Output written", slog.String("outputPath", outputPath))
		}
	}

	slog.Info("Sleeping 2 seconds and then stopping container")

	time.Sleep(2 * time.Second)

	w.Stop(ctx)

	time.Sleep(1 * time.Second)
}

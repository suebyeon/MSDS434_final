package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"

	"github.com/gorilla/mux"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// new task to add from ui
type NewTask struct {
	TaskPriority int     `json:"Task Priority"`
	TaskDuration float64 `json:"Task Duration"`
	TaskDistance int     `json:"Distance to Task in km"`
}

// all previous tasks assigned to technician
type AssignedTask struct {
	TechnicianID string  `json:"Technician ID"`
	TaskPriority int     `json:"Task Priority"`
	TaskDuration float64 `json:"Task Duration"`
	TaskDistance int     `json:"Distance to Task in km"`
}

// new technician assignments for new tasks
type TechAssignment struct {
	TechnicianID string  `json:"Technician ID"`
	TaskPriority int     `json:"Task Priority"`
	TaskDuration float64 `json:"Task Duration"`
	TaskDistance int     `json:"Distance to Task in km"`
}

// ML model output with possibilities of completion for each technician to each task
type ModelOutput struct {
	TechnicianID string  `json:"Technician ID"`
	TaskPriority int     `json:"Task Priority"`
	TaskDuration float64 `json:"Task Duration"`
	TaskDistance int     `json:"Distance to Task in km"`
	Possibility  float64 `json:"probability"`
}

func CreateTaskHandler() http.HandlerFunc {
	return func(rw http.ResponseWriter, r *http.Request) {
		// Read request body
		data, err := io.ReadAll(r.Body)
		if err != nil || len(data) == 0 {
			log.Printf("DEBUG: ReadFile error: %v", err)
			http.Error(rw, "Request body required", http.StatusBadRequest)
			return
		}

		// Parse JSON into Task
		var newtask NewTask
		if err := json.Unmarshal(data, &newtask); err != nil {
			log.Printf("DEBUG: ReadFile error: %v", err)
			http.Error(rw, "Invalid JSON format", http.StatusExpectationFailed)
			return
		}

		// Load existing tasks from file
		var tasks []NewTask
		fileData, err := os.ReadFile("./data/automl_training_data.json")
		if err == nil {
			if err := json.Unmarshal(fileData, &tasks); err != nil {
				log.Printf("DEBUG: ReadFile error: %v", err)
				http.Error(rw, "Error parsing existing tasks", http.StatusInternalServerError)
				return
			}
		} else if !os.IsNotExist(err) {
			log.Printf("DEBUG: ReadFile error: %v", err)
			http.Error(rw, "Unable to read task file", http.StatusInternalServerError)
			return
		}

		// Append new task
		tasks = append(tasks, newtask)

		// Write updated task list to file
		updatedData, err := json.MarshalIndent(tasks, "", "  ")
		if err != nil {
			log.Printf("DEBUG: ReadFile error: %v", err)
			http.Error(rw, "Error encoding task list", http.StatusInternalServerError)
			return
		}
		if err := os.WriteFile("./tasks.json", updatedData, os.ModePerm); err != nil {
			log.Printf("DEBUG: ReadFile error: %v", err)
			http.Error(rw, "Error writing task file", http.StatusInternalServerError)
			return
		}

		rw.WriteHeader(http.StatusCreated)
		rw.Write([]byte("Added New Task"))
	}
}

func ViewAllNewTasksHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		data, err := os.ReadFile("./tasks.json")
		if err != nil {
			log.Printf("DEBUG: Error reading tasks.json: %v", err)
			http.Error(w, "Could not read new tasks", http.StatusInternalServerError)
			return
		}

		var tasks []NewTask
		if err := json.Unmarshal(data, &tasks); err != nil {
			log.Printf("DEBUG: Error parsing tasks.json: %v", err)
			http.Error(w, "Could not parse new tasks", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(tasks)
	}
}

func ViewAssignedTasksHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Read the assigned tasks from file
		data, err := os.ReadFile("./data/automl_training_data.json")
		if err != nil {
			log.Printf("DEBUG: ReadFile error: %v", err)
			http.Error(w, "Failed to read assigned task file", http.StatusInternalServerError)
			return
		}

		var allTasks []AssignedTask
		if err := json.Unmarshal(data, &allTasks); err != nil {
			log.Printf("DEBUG: ReadFile error: %v", err)
			http.Error(w, "Failed to parse task file", http.StatusInternalServerError)
			return
		}

		// Get technicianid from query parameter
		techID := r.URL.Query().Get("technicianid")
		if techID == "" {
			log.Printf("DEBUG: ReadFile error: %v", err)
			http.Error(w, "Missing 'technicianid' query parameter", http.StatusBadRequest)
			return
		}

		// Filter tasks assigned to the specified technician
		var filtered []AssignedTask
		for _, newtask := range allTasks {
			if newtask.TechnicianID == techID {
				filtered = append(filtered, newtask)
			}
		}

		// Return filtered results
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(filtered)
	}
}

func taskKey(priority int, duration float64, distance int) string {
	return fmt.Sprintf("%d-%.2f-%d", priority, duration, distance)
}

func PickBestTech() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		data, err := os.ReadFile("./data/ml_prediction.json")
		if err != nil {
			log.Printf("DEBUG: ReadFile error: %v", err)
			http.Error(w, "Failed to read file", http.StatusInternalServerError)
			return
		}

		var predictions []ModelOutput
		if err := json.Unmarshal(data, &predictions); err != nil {
			log.Printf("DEBUG: ReadFile error: %v", err)
			http.Error(w, "Failed to parse file", http.StatusInternalServerError)
			return
		}

		// Map of taskid to best assignment
		bestAssignments := make(map[string]TechAssignment)
		bestScores := make(map[string]float64)

		for _, p := range predictions {
			key := taskKey(p.TaskPriority, p.TaskDuration, p.TaskDistance)
			if score, exists := bestScores[key]; !exists || p.Possibility > score {
				bestScores[key] = p.Possibility
				bestAssignments[key] = TechAssignment{
					TechnicianID: p.TechnicianID,
					TaskPriority: p.TaskPriority,
					TaskDuration: p.TaskDuration,
					TaskDistance: p.TaskDistance,
				}
			}
		}

		// Collect results into a slice
		var assignments []TechAssignment
		for _, a := range bestAssignments {
			assignments = append(assignments, a)
		}

		// Return filtered results
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(assignments)
	}

}

func homeHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, "Welcome to MSDS 434 Final Project: Technician Task Assignment")
	}
}

func main() {
	// Create new Router
	router := mux.NewRouter()

	// route properly to respective handlers
	router.Handle("/", homeHandler())
	router.Handle("/viewassignmentbytech", ViewAssignedTasksHandler()).Methods("GET")
	router.Handle("/addnewtask", CreateTaskHandler()).Methods("POST")
	router.Handle("/viewallnewtasks", ViewAllNewTasksHandler()).Methods("GET")
	router.Handle("/assigntasktotech", PickBestTech()).Methods("GET")
	router.Handle("/metrics", promhttp.Handler())

	// Create new server and assign the router
	server := http.Server{
		Addr:    ":2112",
		Handler: router,
	}
	fmt.Println("Staring Product Catalog server on Port 2112")
	// Start Server on defined port/host.
	server.ListenAndServe()
}

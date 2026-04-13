package main

import (
	"fmt"
	"os"
)

// NodeInfo matches NodeCapabilities in Python (memopt/vmm/discovery.py)
type NodeInfo struct {
	NodeID     string  `json:"node_id"`
	Host       string  `json:"host"`
	GossipPort int     `json:"gossip_port"`
	Rack       string  `json:"rack"`
	Pod        string  `json:"pod"`
	HBMFreeGB  float64 `json:"hbm_free_gb"`
}

func main() {
	// Not yet implemented.
	// Current discovery uses Redis SCAN — works up to ~10K nodes.
	// Implement this when Redis scan latency exceeds 100ms.
	//
	// Implementation steps:
	// 1. Parse --seed-nodes flag
	// 2. Create memberlist.DefaultLANConfig()
	// 3. Set NodeInfo as memberlist.Delegate metadata
	// 4. memberlist.Create(config) then Join(seedNodes)
	// 5. Listen on /var/run/memopt/discovery.sock
	// 6. On socket request: return JSON list of Members

	fmt.Fprintln(os.Stderr,
		"memopt-discovery: not yet implemented")
	fmt.Fprintln(os.Stderr,
		"Use Redis discovery (set REDIS_URL env var)")
	os.Exit(1)
}

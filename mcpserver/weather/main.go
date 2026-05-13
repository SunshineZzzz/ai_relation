package main

import (
	"context"
	"fmt"
	"os"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

func main() {
	s := server.NewMCPServer(
		"Weather-MCPSvr",
		"1.0.0",
	)

	s.AddTool(mcp.NewTool("get_weather",
		mcp.WithDescription("Get the current weather conditions of the specified city."),
		mcp.WithString("city", mcp.Required(), mcp.Description("The name of the city to get the weather for.")),
	), func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		city, ok := request.GetArguments()["city"].(string)
		if !ok {
			return mcp.NewToolResultError("Invalid city parameter."), nil
		}

		fmt.Fprintf(os.Stderr, "Querying weather for city: %s\n", city)

		// 这里模拟调用 API，你可以换成真正的 http.Get 请求
		weatherInfo := fmt.Sprintf("The weather in %s is: Clear, Temperature 25°C", city)

		return mcp.NewToolResultText(weatherInfo), nil
	})

	// 启动基于 stdio 的服务器
	if err := server.ServeStdio(s); err != nil {
		fmt.Fprintf(os.Stderr, "Server error: %v\n", err)
	}
	// Test:
	// {"jsonrpc":"2.0","method":"tools/call","params":{"name":"get_weather","arguments":{"city":"Beijing"}},"id":1}
}

package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"text/template"

	"github.com/joho/godotenv"
	"github.com/sashabaranov/go-openai"
)

// 获取函数名
func getFunctionName(i interface{}) string {
	fnValue := reflect.ValueOf(i)
	fn := runtime.FuncForPC(fnValue.Pointer())

	if fn == nil {
		return "unknown"
	}

	// 获取原始名称，例如: main.readFile 或 github.com/user/project/pkg.writeFile
	fullName := fn.Name()

	// 找到最后一个 "." 的索引
	// 注意：如果是匿名函数，名称可能包含 "func1" 等后缀，这种处理依然有效
	lastDot := strings.LastIndex(fullName, ".")
	if lastDot != -1 {
		return fullName[lastDot+1:]
	}

	return fullName
}

// 获取OpenRouter API Key
func getAPIKey() string {
	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}

	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		log.Fatal("未找到 OPENROUTER_API_KEY 环境变量，请在 .env 文件中设置。")
	}
	return apiKey
}

// 获取系统代理URL
func getProxyURL() *url.URL {
	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}

	proxyURL := os.Getenv("HTTPS_PROXY")
	if proxyURL == "" {
		log.Fatal("未找到 HTTPS_PROXY 环境变量，请在 .env 文件中设置。")
	}
	parsedURL, err := url.Parse(proxyURL)
	if err != nil {
		log.Fatalf("Error parsing HTTPS_PROXY URL: %v", err)
	}

	return parsedURL
}

type ReActAgent struct {
	Tools            map[string]interface{}
	ToolDocs         map[string]string
	Model            string
	ProjectDirectory string
	Client           *openai.Client
}

func NewReActAgent(tools []interface{}, toolDocs map[string]string, model, projectDirectory string) *ReActAgent {
	toolsMap := make(map[string]interface{})
	for _, tool := range tools {
		toolsMap[getFunctionName(tool)] = tool
	}

	config := openai.DefaultConfig(getAPIKey())

	config.BaseURL = "https://openrouter.ai/api/v1"

	// 创建一个使用系统代理设置的 Transport
	// transport := &http.Transport{
	// 	Proxy: http.ProxyURL(getProxyURL()),
	// }

	// config.HTTPClient = &http.Client{
	// 	Transport: transport,
	// }

	client := openai.NewClientWithConfig(config)

	return &ReActAgent{
		Tools:            toolsMap,
		ToolDocs:         toolDocs,
		Model:            model,
		ProjectDirectory: projectDirectory,
		Client:           client,
	}
}

func (agent *ReActAgent) Run(userInput string) string {
	messages := []map[string]string{
		{"role": "system", "content": agent.renderSystemPrompt(reactSystemPromptTemplate)},
		{"role": "user", "content": fmt.Sprintf("<question>%s</question>", userInput)},
	}

	log.Printf("messages: %v\n\n", messages)

	for {
		content := agent.callModel(messages)

		// Check for thought
		thoughtMatch := regexp.MustCompile(`<thought>(.*?)</thought>`).FindStringSubmatch(content)
		if len(thoughtMatch) > 0 {
			thought := thoughtMatch[1]
			fmt.Printf("\n\n💭 Thought: %s\n", thought)
		}

		// Check for final answer
		if strings.Contains(content, "<final_answer>") {
			finalAnswerMatch := regexp.MustCompile(`<final_answer>(.*?)</final_answer>`).FindStringSubmatch(content)
			if len(finalAnswerMatch) > 0 {
				return finalAnswerMatch[1]
			}
		}

		// Check for action
		actionMatch := regexp.MustCompile(`<action>(.*?)</action>`).FindStringSubmatch(content)
		if len(actionMatch) == 0 {
			log.Fatal("模型未输出 <action>")
		}
		action := actionMatch[1]
		toolName, args := parseAction(action)

		fmt.Printf("\n\n🔧 Action: %s(%s)\n", toolName, strings.Join(args, ", "))
		shouldContinue := "y"
		if toolName == "run_terminal_command" {
			shouldContinue = askUserConfirmation()
		}

		if strings.ToLower(shouldContinue) != "y" {
			return "操作被用户取消"
		}

		// Execute tool
		toolFunc := agent.Tools[toolName]
		if toolFunc != nil {
			fnVal := reflect.ValueOf(toolFunc)
			fnType := fnVal.Type()

			// 获取函数定义的输入参数个数
			numIn := fnType.NumIn()

			// 构造调用参数切片
			callArgs := make([]reflect.Value, numIn)

			for i := 0; i < numIn; i++ {
				// 如果模型提供的 args 足够，则取对应值，否则传零值
				if i < len(args) {
					// 在解析参数时使用
					cleanArg := strings.TrimSpace(args[i])
					unquoted, err := strconv.Unquote(cleanArg)
					if err == nil {
						callArgs[i] = reflect.ValueOf(unquoted)
					} else {
						// 如果不是带引号的字符串，就直接去掉前后的引号
						callArgs[i] = reflect.ValueOf(strings.Trim(cleanArg, "\"'"))
					}
				} else {
					log.Fatalf("模型提供的参数 %d 不足，函数 %s 期望 %d 个参数", i, toolName, numIn)
				}
			}

			// 执行动态调用
			result := fnVal.Call(callArgs)
			observation := result[0].Interface().(string)
			fmt.Printf("\n\n🔍 Observation：%s\n", observation)
			// Append observation back to messages
			messages = append(messages, map[string]string{
				"role":    "user",
				"content": fmt.Sprintf("<observation>%s</observation>", observation),
			})
		}
	}
}

func (agent *ReActAgent) renderSystemPrompt(templateStr string) string {
	tmpl, err := template.New("systemPrompt").Delims("${", "}").Parse(templateStr)
	if err != nil {
		log.Fatal(err)
	}

	var buf strings.Builder
	err = tmpl.Execute(&buf, map[string]interface{}{
		"tool_list":        agent.getToolList(),
		"operating_system": agent.getOperatingSystemName(),
		"current_dir":      agent.ProjectDirectory,
		"file_list":        agent.getFileList(agent.ProjectDirectory),
	})
	if err != nil {
		log.Fatal(err)
	}
	return buf.String()
}

func (agent *ReActAgent) getFileList(dir string) string {
	// 1. 读取目录下的所有文件信息
	entries, err := os.ReadDir(dir)
	if err != nil {
		return "" // 如果读取失败，返回空字符串
	}

	var absPaths []string

	for _, entry := range entries {
		// 2. 拼接成完整路径
		fullPath := filepath.Join(dir, entry.Name())

		// 3. 转换为绝对路径 (相当于 os.path.abspath)
		absPath, err := filepath.Abs(fullPath)
		if err == nil {
			absPaths = append(absPaths, absPath)
		}
	}

	// 4. 用逗号和空格连接起来 (相当于 ", ".join)
	return strings.Join(absPaths, ", ")
}

// 获取操作系统名称
func (agent *ReActAgent) getOperatingSystemName() string {
	// runtime.GOOS 返回的是底层系统的常量名
	// 常见的有: "darwin" (macOS), "windows", "linux", "freebsd" 等
	osMap := map[string]string{
		"darwin":  "macOS",
		"windows": "Windows",
		"linux":   "Linux",
	}

	// 从 map 中获取友好名称
	if name, ok := osMap[runtime.GOOS]; ok {
		return name
	}

	return "Unknown"
}

// 生成工具列表字符串，包含函数签名和简要说明
func (agent *ReActAgent) getToolList() string {
	var toolDescriptions []string

	for name, fn := range agent.Tools {
		t := reflect.TypeOf(fn)
		if t.Kind() != reflect.Func {
			continue
		}

		// 构建签名字符串：func(string, int) string
		ins := []string{}
		for i := 0; i < t.NumIn(); i++ {
			ins = append(ins, t.In(i).String())
		}

		outs := []string{}
		for i := 0; i < t.NumOut(); i++ {
			outs = append(outs, t.Out(i).String())
		}

		signature := fmt.Sprintf("(%s) (%s)", strings.Join(ins, ", "), strings.Join(outs, ", "))

		// 注意：Go 运行时无法获取函数的注释(doc)，只能从你定义的 Map 里取
		doc := agent.ToolDocs[name]
		toolDescriptions = append(toolDescriptions, fmt.Sprintf("- %s%s: %s", name, signature, doc))
	}
	return strings.Join(toolDescriptions, "\n")
}

func (agent *ReActAgent) callModel(messages []map[string]string) string {
	fmt.Println("\n\n正在请求模型，请稍等...")

	// 1. 创建一个正确类型的切片
	apiMessages := make([]openai.ChatCompletionMessage, len(messages))

	// 2. 将 map 里的数据填入结构体
	for i, m := range messages {
		apiMessages[i] = openai.ChatCompletionMessage{
			Role:    m["role"],
			Content: m["content"],
		}
	}

	response, err := agent.Client.CreateChatCompletion(
		context.Background(),
		openai.ChatCompletionRequest{
			Model:    agent.Model,
			Messages: apiMessages,
		},
	)
	if err != nil {
		log.Fatal(err)
	}
	return response.Choices[0].Message.Content
}

func parseAction(codeStr string) (string, []string) {
	re := regexp.MustCompile(`(\w+)\((.*)\)`)
	match := re.FindStringSubmatch(codeStr)
	if len(match) == 0 {
		log.Fatal("Invalid function call syntax")
	}
	funcName := match[1]
	argsStr := match[2]

	// Handle arguments parsing here
	args := strings.Split(argsStr, ",")
	return funcName, args
}

func askUserConfirmation() string {
	reader := bufio.NewReader(os.Stdin)
	fmt.Print("\n\n是否继续？（Y/N）")
	confirmation, _ := reader.ReadString('\n')
	return strings.TrimSpace(confirmation)
}

// Tools functions
// 读取文件内容
func readFile(filePath string) string {
	content, err := os.ReadFile(filePath)
	if err != nil {
		return fmt.Sprintf("读取文件错误: %v", err)
	}
	return string(content)
}

// 写入内容到文件
func writeToFile(filePath, content string) string {
	err := os.WriteFile(filePath, []byte(content), 0644)
	if err != nil {
		return fmt.Sprintf("写入文件错误: %v", err)
	}
	return "写入成功"
}

// 在终端执行命令
func runTerminalCommand(command string) string {
	var cmd *exec.Cmd

	if runtime.GOOS == "windows" {
		// Windows: 使用 cmd /C 执行
		cmd = exec.Command("cmd", "/C", command)
	} else {
		// Linux/macOS: 使用 sh -c 执行
		cmd = exec.Command("sh", "-c", command)
	}

	out, err := cmd.CombinedOutput()
	if err != nil {
		// 注意：在 Agent 中，命令执行失败通常应该返回错误信息给 LLM，
		// 而不是直接 log.Fatal(err) 导致整个程序退出。
		return fmt.Sprintf("执行错误: %v\n输出: %s", err, string(out))
	}
	return string(out)
}

func main() {
	flag.Parse()
	args := flag.Args()

	if len(args) < 1 {
		fmt.Println("错误: 请提供项目目录路径")
		os.Exit(1)
	}

	projectDirectory := args[0]

	info, err := os.Stat(projectDirectory)
	if os.IsNotExist(err) {
		fmt.Printf("错误: 路径 '%s' 不存在\n", projectDirectory)
		os.Exit(1)
	}
	if !info.IsDir() {
		fmt.Printf("错误: '%s' 不是一个目录\n", projectDirectory)
		os.Exit(1)
	}

	projectDir, err := filepath.Abs(projectDirectory)
	if err != nil {
		fmt.Printf("错误: 无法获取绝对路径: %v\n", err)
		os.Exit(1)
	}

	tools := []interface{}{readFile, writeToFile, runTerminalCommand}
	toolDocs := map[string]string{
		"readFile":           "读取文件内容",
		"writeToFile":        "写入内容到文件",
		"runTerminalCommand": "在终端执行命令",
	}

	agent := NewReActAgent(tools, toolDocs, "google/gemma-3-27b-it:free", projectDir)

	// Get the task from the user
	reader := bufio.NewReader(os.Stdin)
	fmt.Print("请输入任务：")
	task, _ := reader.ReadString('\n')

	finalAnswer := agent.Run(task)

	fmt.Printf("\n\n✅ Final Answer：%s\n", finalAnswer)
}

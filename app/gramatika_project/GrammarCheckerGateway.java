import py4j.GatewayServer;  // Py4J gateway
import v4.grammarchecking.threaded.GrammarChecker;  // GrammarChecker class (adjust based on your package)
import java.io.IOException;  // IOException for error handling

public class GrammarCheckerGateway {
    private GrammarChecker grammarChecker;

    public GrammarCheckerGateway() {
        // Initialize the GrammarChecker
        grammarChecker = new GrammarChecker(true, true, 4);
    }

    public String[] getGrammarSuggestions(String sentence) throws IOException, InterruptedException {
        return grammarChecker.getGrammarSuggestions(sentence);
    }

    public static void main(String[] args) {
        GrammarCheckerGateway gateway = new GrammarCheckerGateway();
        GatewayServer server = new GatewayServer(gateway, 25334);  // Use the port you chose
        server.start();
        System.out.println("Gateway Server Started on Port 25334");
    }
}

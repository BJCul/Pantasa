package morphinas_project;
import py4j.GatewayServer;
import java.lang.reflect.Method;  // Added import for Method
import java.lang.reflect.InvocationTargetException;  // Added import for InvocationTargetException

public class LemmatizerGateway {

    private Object stemmerInstance;
    private Method lemmatizeSingleMethod;
    private Method lemmatizeMultipleMethod;

    public LemmatizerGateway() throws Exception {
        // Load the JAR and the Stemmer class
        ClassLoader classLoader = this.getClass().getClassLoader();
        Class<?> stemmerClass = classLoader.loadClass("Stemmer.Stemmer");

        // Create an instance of the Stemmer class
        stemmerInstance = stemmerClass.getDeclaredConstructor().newInstance();

        // Get the lemmatizeSingle and lemmatizeMultiple methods
        lemmatizeSingleMethod = stemmerClass.getMethod("lemmatizeSingle", String.class);
        lemmatizeMultipleMethod = stemmerClass.getMethod("lemmatizeMultiple", String[].class);
    }

    public String lemmatizeSingle(String word) throws InvocationTargetException, IllegalAccessException {
        // Call the lemmatizeSingle method using reflection
        return (String) lemmatizeSingleMethod.invoke(stemmerInstance, word);
    }

    public String[] lemmatizeMultiple(String[] words) throws InvocationTargetException, IllegalAccessException {
        // Call the lemmatizeMultiple method using reflection
        return (String[]) lemmatizeMultipleMethod.invoke(stemmerInstance, (Object) words);
    }

    public static void main(String[] args) throws Exception {
        LemmatizerGateway app = new LemmatizerGateway();
        GatewayServer server = new GatewayServer(app);
        server.start();
        System.out.println("Gateway Server Started");
    }
}

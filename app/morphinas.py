from py4j.java_gateway import JavaGateway, GatewayParameters, launch_gateway

# Initialize Morphinas
def initialize_morphinas(jar_path):
    port = launch_gateway(classpath=jar_path, die_on_exit=True)
    gateway = JavaGateway(gateway_parameters=GatewayParameters(port=port))
    morphinas = gateway.jvm.com.example.morphinas.Morphinas()  # Update the class path
    return morphinas

# Use Morphinas to lemmatize text
def lemmatize_with_morphinas(text, morphinas):
    return morphinas.lemmatize(text)

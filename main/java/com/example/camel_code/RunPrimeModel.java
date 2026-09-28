package com.example.camel_code;

import ai.onnxruntime.*;

import java.nio.LongBuffer;
import java.util.Collections;

public class RunPrimeModel {

    public static void main(String[] args) throws Exception {

        // ONNX Runtime environment
        OrtEnvironment environment =
                OrtEnvironment.getEnvironment();

        // Load ONNX model
        OrtSession.SessionOptions options =
                new OrtSession.SessionOptions();
        System.out.println(
                "ONNX Runtime version = "
                        + OrtEnvironment.getEnvironment().getVersion()
        );
        OrtSession session =
                environment.createSession(
                        "src/main/resources/prime_math_model.onnx",
                        options);

        System.out.println("ONNX model loaded!");

        // Number to test
        int number = 101;

        // ONNX input must be INT64
        long[] inputData = {number};

        OnnxTensor inputTensor =
                OnnxTensor.createTensor(
                        environment,
                        LongBuffer.wrap(inputData),
                        new long[]{1});

        // Run model
        OrtSession.Result result =
                session.run(
                        Collections.singletonMap(
                                "number",
                                inputTensor));

        // Read output
        long[] output =
                (long[]) result.get(0).getValue();

        long prediction = output[0];

        String answer =
                prediction == 1
                        ? "PRIME"
                        : "NOT PRIME";

        System.out.println(number + " => " + answer);

        // Cleanup
        inputTensor.close();
        result.close();
        session.close();
        options.close();
    }
}
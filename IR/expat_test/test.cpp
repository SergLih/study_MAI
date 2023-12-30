#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"
#include "vector.hpp"
#include "tstring.h"


SCENARIO( "vectors should expand", "[tvector]" ) {

    GIVEN("An empty vector") {
        TVector<int> v;

        REQUIRE(v.size() == 0);
        REQUIRE(v.capacity() == VECTOR_DEFAULT_CAPACITY);

        WHEN("one element is added") {
            v.push(10);

            THEN("the size changes and capacity does not") {
                REQUIRE(v.size() == 1);
                REQUIRE(v.capacity() == VECTOR_DEFAULT_CAPACITY);
            }
        }
        WHEN("two elements are added") {
            v.push(10);
            v.push(20);

            THEN("the size and capacity change") {
                REQUIRE(v.size() == 2);
                REQUIRE(v.capacity() == int(VECTOR_DEFAULT_CAPACITY * VECTOR_EXTENSION_FACTOR));
            }
        }
    }
}
SCENARIO( "vectors should (de)serialize", "[tvector][serialization][tstring]" ) {
    GIVEN("One vector with three TStrings") {
        TVector<TString> v1, v2;
        v1.push(TString("abc"));
        v1.push(TString("defgh"));
        v1.push(TString("xyz"));
        REQUIRE(v1.size() == 3);
        REQUIRE(v1.capacity() == 4);
        WHEN("compared to other vector with same TStrings") {
            v2.push(TString("abc"));
            v2.push(TString("defgh"));
            v2.push(TString("xyz"));

            THEN("they are equal") {
                REQUIRE(v1 == v2);
            }
        }

        WHEN("serialized to file and deserialized back to other vector") {
            string filename = "test_vector_of_strings.bin";
            ofstream file_out(filename, ios::binary);
            v1.serialize(file_out);
            file_out.close();

            ifstream file_in(filename, ios::binary);
            v2.deserialize(file_in);
            file_in.close();

            THEN("they are equal") {
                REQUIRE(v1 == v2);
            }
        }
    }
    GIVEN("One vector with three value-type values") {
        TVector<float> v1, v2;
        v1.push(1.2);
        v1.push(1.4);
        v1.push(1.6);
        REQUIRE(v1.size() == 3);
        REQUIRE(v1.capacity() == 4);
        WHEN("compared to other vector with same values") {
            v2.push(1.2);
            v2.push(1.4);
            v2.push(1.6);

            THEN("they are equal") {
                REQUIRE(v1 == v2);
            }
        }

        WHEN("serialized to file and deserialized back to other vector") {
            string filename = "test_vector_of_numbers.bin";
            ofstream file_out(filename, ios::binary);
            v1.serialize(file_out);
            file_out.close();

            ifstream file_in(filename, ios::binary);
            v2.deserialize(file_in);
            file_in.close();

            THEN("they are equal") {
                REQUIRE(v1 == v2);
            }
        }
    }
}
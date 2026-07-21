#include <errno.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

extern char **environ;

static int report_failure(const char *message) {
    dprintf(STDERR_FILENO, "CloudPoint runtime launcher: %s\n", message);
    return 126;
}

int main(int argument_count, char **arguments) {
    char unresolved_path[PATH_MAX];
    uint32_t unresolved_size = sizeof(unresolved_path);
    if (_NSGetExecutablePath(unresolved_path, &unresolved_size) != 0) {
        return report_failure("executable path is too long");
    }

    char resolved_path[PATH_MAX];
    if (realpath(unresolved_path, resolved_path) == NULL) {
        return report_failure("could not resolve executable path");
    }

    char *separator = strrchr(resolved_path, '/');
    if (separator == NULL || separator[1] == '\0') {
        return report_failure("executable path is malformed");
    }
    const char *program_name = separator + 1;
    const char *module = NULL;
    if (strcmp(program_name, "cloudpoint-worker") == 0) {
        module = "cloudpoint_worker.cli";
    } else if (strcmp(program_name, "cloudpoint-model") == 0) {
        module = "cloudpoint_worker.model_prep.cli";
    } else {
        return report_failure("executable name is not recognized");
    }

    *separator = '\0';
    char python_path[PATH_MAX];
    int path_length = snprintf(
        python_path,
        sizeof(python_path),
        "%s/python3.12",
        resolved_path
    );
    if (path_length < 0 || (size_t)path_length >= sizeof(python_path)) {
        return report_failure("Python path is too long");
    }

    size_t child_count = (size_t)argument_count + 5;
    char **child_arguments = calloc(child_count, sizeof(char *));
    if (child_arguments == NULL) {
        return report_failure("could not allocate arguments");
    }
    child_arguments[0] = python_path;
    child_arguments[1] = "-I";
    child_arguments[2] = "-B";
    child_arguments[3] = "-m";
    child_arguments[4] = (char *)module;
    for (int index = 1; index < argument_count; index++) {
        child_arguments[index + 4] = arguments[index];
    }
    child_arguments[argument_count + 4] = NULL;

    execve(python_path, child_arguments, environ);
    int failure = errno;
    free(child_arguments);
    dprintf(
        STDERR_FILENO,
        "CloudPoint runtime launcher: exec failed: %s\n",
        strerror(failure)
    );
    return 126;
}

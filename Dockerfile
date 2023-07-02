ARG PY_VERSION=3.10

FROM python:3.10 AS build-3.10
WORKDIR /work
COPY tsax tsax
COPY setup.py pyproject.toml README.md LICENSE mypy.ini .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install jax[cpu] .[all] && \
    mypy -p tsax && \
    rm -rf tsax && \
    rm setup.py pyproject.toml README.md LICENSE mypy.ini


FROM build-3.10 AS test-3.10
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install coverage unittest-xml-reporting
COPY test test
WORKDIR /work/test
COPY .coveragerc .
RUN coverage run --source tsax -m xmlrunner discover || true
RUN mkdir -p /coverage && cp -v .coverage.* /coverage && \
    mkdir -p /unittest && cp -v *.xml /unittest


FROM python:3.10 AS coverage
WORKDIR /coverage
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install coverage
COPY tsax tsax
COPY .coveragerc .
COPY --from=test-3.10 /coverage/* .
RUN coverage combine && \
    echo "## Test Coverage\n\`\`\`\n" >> summary.md && \
    coverage report | tee -a summary.md && \
    echo "\n\`\`\`" >> summary.md && \
    mkdir -p /coverage/html && coverage html -d /coverage/html


FROM python:3.10 AS doc
WORKDIR /ci
RUN --mount=type=cache,target=/var/lib/apt/lists \
    apt update && apt -y --no-install-recommends install graphviz
RUN --mount=type=cache,target=/root/.cache/pip pip install \
    sphinx \
    furo \
    sphinx-automodapi \
    myst-parser
COPY setup.py pyproject.toml README.md LICENSE .
COPY tsax tsax
RUN --mount=type=cache,target=/root/.cache/pip pip install .[doc]
COPY doc doc
COPY example example
RUN sphinx-build -W -b html doc /html


FROM scratch AS CI
COPY --from=test-3.10 /unittest/* /unittest/
COPY --from=coverage /coverage/* /coverage/
COPY --from=doc /html /html
CMD [""]


FROM build-${PY_VERSION} AS tsax


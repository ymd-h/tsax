ARG PY_VERSION=3.10

FROM scratch AS test
WORKDIR /test
COPY setup.py pyproject.toml README.md LICENSE mypy.ini .coveragerc ci.sh .
COPY tsax tsax
COPY test test

FROM python:3.8 AS test-3.8
WORKDIR /work
COPY --from=test /test .
RUN --mount=type=cache,target=/root/.cache/pip ./ci.sh

FROM python:3.9 AS test-3.9
WORKDIR /work
COPY --from=test /test .
RUN --mount=type=cache,target=/root/.cache/pip ./ci.sh

FROM python:3.10 AS test-3.10
WORKDIR /work
COPY --from=test /test .
RUN --mount=type=cache,target=/root/.cache/pip ./ci.sh

FROM python:3.11 AS test-3.11
WORKDIR /work
COPY --from=test /test .
RUN --mount=type=cache,target=/root/.cache/pip ./ci.sh


FROM python:3.10 AS coverage
WORKDIR /coverage
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install coverage
COPY tsax tsax
COPY .coveragerc .
COPY --from=test-3.8 /coverage/* .
COPY --from=test-3.9 /coverage/* .
COPY --from=test-3.10 /coverage/* .
COPY --from=test-3.11 /coverage/* .
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
RUN --mount=type=cache,target=/root/.cache/pip pip install "jax[cpu]<=0.4.10" .[all]
COPY doc doc
COPY example example
RUN sphinx-build -v -W -b html doc /html


FROM python:3.10 AS wheel
WORKDIR /build
COPY tsax tsax
COPY setup.py pyproject.toml README.md LICENSE mypy.ini .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel /build -w /dist --no-deps


FROM scratch AS CI
COPY --from=test-3.10 /unittest/* /unittest/
COPY --from=coverage /coverage/* /coverage/
COPY --from=doc /html /html
COPY --from=wheel /dist /dist
CMD [""]


FROM python:${PY_VERSION} AS tsax
WORKDIR /work
COPY tsax tsax
COPY setup.py pyproject.toml README.md LICENSE .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "jax[cpu]<=0.4.10" .[cli,board] && \
    rm -rf tsax && \
    rm setup.py pyproject.toml README.md LICENSE

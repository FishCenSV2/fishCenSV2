# Server and Session
This documentation contains breakdowns of some (not all) of the functions used in the `Server` class as well the `Session` class. In the code below you might see ASIO functions that are asynchronous (look for keyword async). These functions are asynchronous in the sense that they immediately return and the lambda functions passed into it will be called whenever. The lambda may outlive its caller as a result.

## Table of Contents

- [Server and Session](#server-and-session)
  - [Table of Contents](#table-of-contents)
  - [Server Class](#server-class)
    - [Constructor](#constructor)
    - [Accepting Connection](#accepting-connection)
  - [Session Class](#session-class)
    - [Reading](#reading)
    - [Writing](#writing)

## Server Class
The main idea behind the server is to asynchronously accept multiple connections from multiple clients. A connection is represented by a session object. 

### Constructor
--- 
The following shows the `Server` constructor.

```cpp
Server::Server(boost::asio::io_context& io_context, unsigned short port, std::mutex& mutex)
    : _io_context(io_context), _acceptor(io_context, tcp::endpoint(tcp::v4(), port)), mutex(mutex) {
    do_accept();

}
```

The presence of the mutex is explained later in the section [Session](#session-class). The constructor immediately calls `do_accept()` but it still won't run. The user **must** call the `run` method after this.

### Accepting Connection
---
The following shows the `do_accept` method.

```cpp
void Server::do_accept() {
    _acceptor.async_accept([this](boost::system::error_code ec, tcp::socket socket) {
        if (!ec) {
            std::cout << "Connection accepted. Creating session\n";

            std::make_shared<Session>(std::move(socket), *this)->start();
        }

        do_accept();
        });
}
```

The `_acceptor` object asynchronously accepts a connection and calls `do_accept()` to accept another connection. The callback lambda, when invoked, creates a shared pointer for a `Session` object. It also passes the socket and server object itself to the constructor of the `Session` object. Next it immediately calls the `start` method of `Session`. 

One may wonder why the `Session` object won't be destroyed when the lambda goes out of scope as no other shared pointers point to it. What isn't shown is that the `Session` object actually creates a shared pointer to itself which means the object will continue to live. More detail can be found in the section [Session](#session-class)

## Session Class
This class represents a connection between a client and a server. It is an asynchronous loop of reading from and writing to the client. An important thing to note is that `Session` inherits from the following

```cpp
class Session : public std::enable_shared_from_this<Session> {
    //Code here...
}
```

The following description is grabbed from the [C++ Reference](https://en.cppreference.com/w/cpp/memory/enable_shared_from_this)

> `std::enable_shared_from_this` allows an object `t` that is currently managed by a `std::shared_ptr` named `pt` to safely generate additional `std::shared_ptr` instances `pt1`, `pt2`, ... that all share ownership of `t` with `pt`.
Publicly inheriting from `std::enable_shared_from_this<T>` provides the type `T` with a member function shared_from_this. If an object `t` of type `T` is managed by a `std::shared_ptr<T>` named `pt`, then calling `T::shared_from_this` will return a new `std::shared_ptr<T>` that shares ownership of `t` with `pt`.

We will explain its purpose in the [Reading](#reading) section. I did not add this to the appendix since it only appears in this one class and it would be harder to digest seeing it from the appendix only.

### Reading
---
Before we talk about the `do_read` method we should recall how the `do_accept` method of the `Server` class in the previous section creates a shared pointer of `Session` and immediately calls its method called `start`. This method is implemented as follows

```cpp
void Session::start() {
    do_read();
}
```

So this immediately calls `do_read()` which is shown below

```cpp
void Session::do_read() {

    //Get shared pointer to Session object
    auto self(shared_from_this());

    boost::asio::async_read(_socket, boost::asio::buffer(&request, sizeof(request)),
        [this, self](boost::system::error_code ec, std::size_t len) {
            if (!ec) {
                do_write();
            }

            else {
                std::cout << ec.message() << "\n";
            }
        });
   
}
```

The call to `shared_from_this()` creates a shared pointer to `this` which prevents our `Session` object from being destroyed as we had thought in the [Accepting Connection](#accepting-connection) section. We then asynchronously read a 4 byte integer from the socket which is the user requesting counting data. Obviously, reading any 4 byte integer seems arbitrary, but the idea was that later on the user might request different data such as temperature. Finally, the lambda callback will then call the `do_write` method.

### Writing
---
The following shows the `do_write` method.

```cpp
void Session::do_write() {

    //Get shared pointer to Session object
    auto self(shared_from_this());

    {
        std::lock_guard<std::mutex> guard(server.mutex);

        /*
        The thread function in main modifies the buffer in a way that invalidates all the
        iterators for all references. So we resort to a copy to a member variable. This must
        be a member variable since boost::asio::buffer does not own the object passed in to it
        meaning we must guarantee its lifetime ourselves.
        */
        data_cpy = server.data;
    }

    boost::asio::async_write(_socket, boost::asio::buffer(data_cpy),
        [this, self](boost::system::error_code ec, std::size_t len) {
            if (!ec) {
                do_read();
            }

            else {
                std::cout << ec.message() << "\n";
            }

        });
}   
```

Once again we create a shared pointer to `this`. Recall that the `Server` class had a mutex. This mutex is used to create a copy of the counting data vector from the server. The comment above in the code explains the need for copying. It may be possible to rewrite the code in `main.cpp` to only modify the vectors elements instead of a whole new assignment (e.g. `std::vector<int> v = {x,y};` vs `v[0] += 1; v[1] += 1;`). The reason it is currently like this is because the `Server` class was originally made for sending image frames which must reassign the whole vector. We still must guarantee the alternative way does not cause a runtime error as `boost::asio::buffer()` does not own the object passed into it.

Once the lambda callback is called it immediately calls `do_read()` which, as we know, calls `do_write()` leading to an infinite loop of reading and writing.
